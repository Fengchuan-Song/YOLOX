#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import contextlib
import copy
import io
import itertools
import json
import os
import random
import tempfile
import time
import warnings
import xml.etree.ElementTree as ET
from collections import ChainMap, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from loguru import logger
from pycocotools.coco import COCO
from tqdm import tqdm

from exps.default.yolox_nano import Exp as NanoExp
from exps.default.yolox_s import Exp as SmallExp
from yolox.core import Trainer, launch
from yolox.data import (
    DataLoader,
    InfiniteSampler,
    MosaicDetection,
    TrainTransform,
    ValTransform,
    YoloBatchSampler,
    worker_init_reset_seed,
)
from yolox.data.datasets.coco import remove_useless_info
from yolox.data.datasets.datasets_wrapper import CacheDataset, cache_read_img
from yolox.evaluators import COCOEvaluator
from yolox.exp import check_exp_value
from yolox.utils import (
    adjust_status,
    all_reduce_norm,
    configure_module,
    configure_nccl,
    configure_omp,
    gather,
    get_num_devices,
    get_rank,
    is_main_process,
    is_parallel,
    postprocess,
    save_checkpoint,
    synchronize,
    time_synchronized,
    wait_for_the_master,
    xyxy2xywh,
)


DEFAULT_DATASET_ROOT = '/data_ssd/datasets/WaterScenes'
DEFAULT_CLASSES = ("ship",)
MODEL_SPECS = {
    "nano": ("yolox_nano_waterscenes", NanoExp),
    "s": ("yolox_s_waterscenes", SmallExp),
}


def make_parser():
    parser = argparse.ArgumentParser("Train YOLOX on WaterScenes")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--models", nargs="+", default=["nano", "s"], choices=MODEL_SPECS.keys())
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    parser.add_argument("--output-dir", default="YOLOX_outputs")
    parser.add_argument("--ann-cache-dir", default=os.path.join("datasets", "WaterScenes", "annotations"))
    parser.add_argument("--force-rebuild-ann", action="store_true", help="Rebuild cached COCO json annotations.")
    parser.add_argument("--max-epoch", type=int, default=None)
    parser.add_argument("--batch-size", "-b", type=int, default=16)
    parser.add_argument("--devices", "-d", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", choices=["ram", "disk"])
    parser.add_argument("--occupy", "-o", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt", "-c", default=None)
    parser.add_argument("--start_epoch", "-e", type=int, default=None)
    parser.add_argument("--dist-backend", default="nccl")
    parser.add_argument("--dist-url", default=None)
    parser.add_argument("--num_machines", type=int, default=1)
    parser.add_argument("--machine_rank", type=int, default=0)
    parser.add_argument("--logger", default="wandb", choices=["wandb", "tensorboard", "mlflow"])
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Extra YOLOX/W&B options, e.g. wandb-project WaterScenes wandb-name nano",
    )
    return parser


def _split_image_paths(dataset_root, split):
    split_file = Path(dataset_root) / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"split file not found: {split_file}")

    image_paths = []
    for line in split_file.read_text(encoding="utf-8").splitlines():
        line = line.strip().replace("\\", "/")
        if not line:
            continue
        if line.startswith("./"):
            line = line[2:]
        image_paths.append(line)
    return image_paths


def _read_image_size(dataset_root, rel_image):
    image_file = Path(dataset_root) / rel_image
    img = cv2.imread(str(image_file))
    if img is None:
        raise FileNotFoundError(f"image file not found or unreadable: {image_file}")
    height, width = img.shape[:2]
    return width, height


def _xml_to_coco_objects(xml_file, class_to_id):
    if not xml_file.exists():
        return None, []

    root = ET.parse(xml_file).getroot()
    size = root.find("size")
    width = int(float(size.findtext("width")))
    height = int(float(size.findtext("height")))
    objects = []

    for obj in root.iter("object"):
        name = obj.findtext("name", default="").strip()
        if name not in class_to_id:
            continue
        difficult = int(float(obj.findtext("difficult", default="0")))
        bbox = obj.find("bndbox")
        xmin = max(0.0, float(bbox.findtext("xmin")))
        ymin = max(0.0, float(bbox.findtext("ymin")))
        xmax = min(float(width), float(bbox.findtext("xmax")))
        ymax = min(float(height), float(bbox.findtext("ymax")))
        box_w = max(0.0, xmax - xmin)
        box_h = max(0.0, ymax - ymin)
        if box_w <= 0 or box_h <= 0:
            continue
        objects.append(
            {
                "category_id": class_to_id[name],
                "bbox": [xmin, ymin, box_w, box_h],
                "area": box_w * box_h,
                "iscrowd": 0,
                "ignore": difficult,
            }
        )
    return (width, height), objects


def build_coco_annotations(dataset_root, splits, classes, ann_cache_dir, force=False):
    dataset_root = Path(dataset_root)
    ann_cache_dir = Path(ann_cache_dir)
    ann_cache_dir.mkdir(parents=True, exist_ok=True)
    class_to_id = {name: idx + 1 for idx, name in enumerate(classes)}
    categories = [{"id": idx + 1, "name": name} for idx, name in enumerate(classes)]
    ann_files = {}

    for split in splits:
        out_file = ann_cache_dir / f"waterscenes_{split}.json"
        ann_files[split] = str(out_file.resolve())
        if out_file.exists() and not force:
            continue

        images = []
        annotations = []
        ann_id = 1
        image_id = 1
        skipped_missing_xml = 0
        for rel_image in _split_image_paths(dataset_root, split):
            stem = Path(rel_image).stem
            xml_file = dataset_root / "detection" / "xml_modified" / f"{stem}.xml"
            size, objects = _xml_to_coco_objects(xml_file, class_to_id)
            if size is None:
                skipped_missing_xml += 1
                continue
            width, height = size

            images.append(
                {
                    "id": image_id,
                    "file_name": rel_image,
                    "width": width,
                    "height": height,
                }
            )
            for obj in objects:
                obj["id"] = ann_id
                obj["image_id"] = image_id
                annotations.append(obj)
                ann_id += 1
            image_id += 1

        dataset = {
            "info": {"description": "WaterScenes converted from VOC XML"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        out_file.write_text(json.dumps(dataset), encoding="utf-8")
        logger.info(
            "Wrote {} images and {} boxes to {}; skipped {} images without XML",
            len(images),
            len(annotations),
            out_file,
            skipped_missing_xml,
        )
    return ann_files


class WaterScenesCOCODataset(CacheDataset):
    def __init__(
        self,
        data_dir,
        json_file,
        img_size=(640, 640),
        preproc=None,
        cache=False,
        cache_type="ram",
        name="waterscenes",
    ):
        self.data_dir = data_dir
        self.json_file = json_file
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.coco = COCO(json_file)
        remove_useless_info(self.coco)
        self.ids = self.coco.getImgIds()
        self.num_imgs = len(self.ids)
        self.class_ids = sorted(self.coco.getCatIds())
        self.cats = self.coco.loadCats(self.class_ids)
        self._classes = tuple(c["name"] for c in self.cats)
        self.annotations = self._load_coco_annotations()
        path_filename = [anno[3] for anno in self.annotations]
        super().__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=data_dir,
            cache_dir_name=f"cache_{name}",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type,
        )

    def __len__(self):
        return self.num_imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_id) for _id in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        res = np.zeros((len(objs), 5))
        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r
        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))
        return res, img_info, resized_info, im_ann["file_name"]

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_image(self, index):
        file_name = self.annotations[index][3]
        img_file = os.path.join(self.data_dir, file_name)
        img = cv2.imread(img_file)
        assert img is not None, f"file named {img_file} not found"
        return img

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        return cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

    @cache_read_img(use_cache=True)
    def read_img(self, index):
        return self.load_resized_img(index)

    def pull_item(self, index):
        id_ = self.ids[index]
        label, origin_image_size, _, _ = self.annotations[index]
        img = self.read_img(index)
        return img, copy.deepcopy(label), origin_image_size, np.array([id_])

    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index)
        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id


class WaterScenesEvaluator(COCOEvaluator):
    def evaluate(self, model, distributed=False, half=False, trt_file=None, decoder=None, test_size=None, return_outputs=False):
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        data_list = []
        output_data = defaultdict()
        progress_bar = tqdm if is_main_process() else iter
        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(progress_bar(self.dataloader)):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()
                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start
                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end
            data_list_elem, image_wise_data = self.convert_to_coco_format(outputs, info_imgs, ids, return_outputs=True)
            data_list.extend(data_list_elem)
            output_data.update(image_wise_data)

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            synchronize()
            data_list = gather(data_list, dst=0)
            output_data = gather(output_data, dst=0)
            data_list = list(itertools.chain(*data_list))
            output_data = dict(ChainMap(*output_data))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        if return_outputs:
            return eval_results, output_data
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids, return_outputs=False):
        data_list = []
        image_wise_data = defaultdict(dict)
        for output, img_h, img_w, img_id in zip(outputs, info_imgs[0], info_imgs[1], ids):
            if output is None:
                continue
            output = output.cpu()
            bboxes = output[:, 0:4]
            scale = min(self.img_size[0] / float(img_h), self.img_size[1] / float(img_w))
            bboxes /= scale
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            image_wise_data.update(
                {
                    int(img_id): {
                        "bboxes": [box.numpy().tolist() for box in bboxes],
                        "scores": [score.numpy().item() for score in scores],
                        "categories": [self.dataloader.dataset.class_ids[int(cls[ind])] for ind in range(bboxes.shape[0])],
                    }
                }
            )
            bboxes = xyxy2xywh(bboxes)
            for ind in range(bboxes.shape[0]):
                data_list.append(
                    {
                        "image_id": int(img_id),
                        "category_id": self.dataloader.dataset.class_ids[int(cls[ind])],
                        "bbox": bboxes[ind].numpy().tolist(),
                        "score": scores[ind].numpy().item(),
                        "segmentation": [],
                    }
                )
        if return_outputs:
            return data_list, image_wise_data
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            self.latest_metrics = {}
            return 0, 0, None

        logger.info("Evaluate in main process...")
        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()
        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)
        info = (
            "Average forward time: {:.2f} ms, Average NMS time: {:.2f} ms, "
            "Average inference time: {:.2f} ms\n"
        ).format(a_infer_time, a_nms_time, a_infer_time + a_nms_time)

        if len(data_dict) == 0:
            self.latest_metrics = {"map50": 0.0, "map75": 0.0, "map50_95": 0.0, "mar50_95": 0.0}
            return 0, 0, info

        coco_gt = self.dataloader.dataset.coco
        _, tmp = tempfile.mkstemp()
        json.dump(data_dict, open(tmp, "w"))
        coco_dt = coco_gt.loadRes(tmp)
        try:
            from yolox.layers import COCOeval_opt as COCOeval
        except ImportError:
            from pycocotools.cocoeval import COCOeval
            logger.warning("Use standard COCOeval.")

        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            coco_eval.summarize()
        info += redirect_string.getvalue()

        stats = coco_eval.stats
        self.latest_metrics = {
            "map50": float(stats[1]),
            "map75": float(stats[2]),
            "map50_95": float(stats[0]),
            "mar50_95": float(stats[8]),
        }
        return self.latest_metrics["map50_95"], self.latest_metrics["map50"], info


class WaterScenesTrainer(Trainer):
    def evaluate_and_save_model(self):
        evalmodel = self.ema_model.ema if self.use_model_ema else self.model
        if is_parallel(evalmodel):
            evalmodel = evalmodel.module

        with adjust_status(evalmodel, training=False):
            (ap50_95, ap50, summary), predictions = self.exp.eval(
                evalmodel, self.evaluator, self.is_distributed, return_outputs=True
            )

        metrics = getattr(self.evaluator, "latest_metrics", {})
        update_best_ckpt = ap50_95 > self.best_ap
        self.best_ap = max(self.best_ap, ap50_95)

        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger.add_scalar("test/map50", metrics.get("map50", ap50), self.epoch + 1)
                self.tblogger.add_scalar("test/map75", metrics.get("map75", 0.0), self.epoch + 1)
                self.tblogger.add_scalar("test/map50_95", metrics.get("map50_95", ap50_95), self.epoch + 1)
                self.tblogger.add_scalar("test/mar50_95", metrics.get("mar50_95", 0.0), self.epoch + 1)
            elif self.args.logger == "wandb":
                self.wandb_logger.log_metrics(
                    {
                        "test/map50": metrics.get("map50", ap50),
                        "test/map75": metrics.get("map75", 0.0),
                        "test/map50_95": metrics.get("map50_95", ap50_95),
                        "test/mar50_95": metrics.get("mar50_95", 0.0),
                        "train/epoch": self.epoch + 1,
                    }
                )
                self.wandb_logger.log_images(predictions)
            elif self.args.logger == "mlflow":
                logs = {
                    "test/map50": metrics.get("map50", ap50),
                    "test/map75": metrics.get("map75", 0.0),
                    "test/map50_95": metrics.get("map50_95", ap50_95),
                    "test/mar50_95": metrics.get("mar50_95", 0.0),
                    "train/epoch": self.epoch + 1,
                }
                self.mlflow_logger.on_log(self.args, self.exp, self.epoch + 1, logs)
            logger.info("\n" + summary)
        synchronize()

        self.save_ckpt("last_epoch", update_best_ckpt, ap=ap50_95)
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}", ap=ap50_95)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False, ap=None):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}", self.file_name)
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_ap": self.best_ap,
                "curr_ap": ap,
            }
            save_checkpoint(ckpt_state, update_best_ckpt, self.file_name, ckpt_name)
            if self.args.logger == "wandb":
                self.wandb_logger.save_checkpoint(
                    self.file_name,
                    ckpt_name,
                    update_best_ckpt,
                    metadata={
                        "epoch": self.epoch + 1,
                        "optimizer": self.optimizer.state_dict(),
                        "best_ap": self.best_ap,
                        "curr_ap": ap,
                    },
                )


class WaterScenesExpMixin:
    def configure_waterscenes(self, args, model_key, ann_files):
        exp_name, _ = MODEL_SPECS[model_key]
        self.exp_name = exp_name
        self.output_dir = args.output_dir
        self.num_classes = len(args.classes)
        self.data_dir = args.dataset_root
        self.train_json = ann_files["train"]
        self.test_json = ann_files["test"]
        self.data_num_workers = args.num_workers
        self.eval_interval = 1
        if args.max_epoch is not None:
            self.max_epoch = args.max_epoch

    def get_dataset(self, cache=False, cache_type="ram"):
        return WaterScenesCOCODataset(
            data_dir=self.data_dir,
            json_file=self.train_json,
            img_size=self.input_size,
            preproc=TrainTransform(max_labels=120, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
            cache=cache,
            cache_type=cache_type,
            name=f"{self.exp_name}_train",
        )

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=None):
        if self.dataset is None:
            with wait_for_the_master():
                assert cache_img is None, "cache_img must be None if dataset was not pre-created"
                self.dataset = self.get_dataset(cache=False, cache_type=cache_img)

        self.dataset = MosaicDetection(
            dataset=self.dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(max_labels=120, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)
        batch_sampler = YoloBatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False, mosaic=not no_aug)
        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "batch_sampler": batch_sampler,
            "worker_init_fn": worker_init_reset_seed,
        }
        return DataLoader(self.dataset, **dataloader_kwargs)

    def get_eval_dataset(self, **kwargs):
        legacy = kwargs.get("legacy", False)
        return WaterScenesCOCODataset(
            data_dir=self.data_dir,
            json_file=self.test_json,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
            cache=False,
            name=f"{self.exp_name}_test",
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        return WaterScenesEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed, testdev=testdev, legacy=legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )

    def get_trainer(self, args):
        return WaterScenesTrainer(self, args)


class WaterScenesNanoExp(WaterScenesExpMixin, NanoExp):
    pass


class WaterScenesSmallExp(WaterScenesExpMixin, SmallExp):
    pass


def build_exp(args, model_key, ann_files):
    exp_cls = WaterScenesNanoExp if model_key == "nano" else WaterScenesSmallExp
    exp = exp_cls()
    exp.configure_waterscenes(args, model_key, ann_files)
    return exp


@logger.catch
def main(exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn("Seeded training enables deterministic CUDNN and can slow training.")

    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    trainer = exp.get_trainer(args)
    trainer.train()


def train_one_model(args, model_key, ann_files):
    exp = build_exp(args, model_key, ann_files)
    check_exp_value(exp)
    args.experiment_name = exp.exp_name

    if not args.resume:
        args.ckpt = None

    if args.cache is not None:
        exp.dataset = exp.get_dataset(cache=True, cache_type=args.cache)

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()
    dist_url = "auto" if args.dist_url is None else args.dist_url
    logger.info("Starting {} from scratch on WaterScenes", exp.exp_name)
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )


if __name__ == "__main__":
    configure_module()
    args = make_parser().parse_args()
    ann_files = build_coco_annotations(
        dataset_root=args.dataset_root,
        splits=("train", "test"),
        classes=args.classes,
        ann_cache_dir=args.ann_cache_dir,
        force=args.force_rebuild_ann,
    )

    for model_key in args.models:
        model_args = copy.deepcopy(args)
        train_one_model(model_args, model_key, ann_files)
