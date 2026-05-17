#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Convert YOLO txt annotations to COCO JSON annotations.

YOLO txt format per line:
    class_id x_center y_center width height

The box values are normalized to [0, 1]. COCO uses:
    [x_min, y_min, width, height]
in pixel coordinates.
"""

import argparse
import json
from pathlib import Path

import cv2


IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def make_parser():
    parser = argparse.ArgumentParser("YOLO txt to COCO json converter")
    parser.add_argument(
        "--images-dir",
        required=True,
        type=Path,
        help="directory containing images, for example datasets/mydata/train2017",
    )
    parser.add_argument(
        "--labels-dir",
        required=True,
        type=Path,
        help="directory containing YOLO txt labels, for example datasets/mydata/labels/train2017",
    )
    parser.add_argument(
        "--classes",
        required=True,
        type=Path,
        help="txt file with one class name per line",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="output COCO json path, for example datasets/mydata/annotations/instances_train2017.json",
    )
    parser.add_argument(
        "--start-image-id",
        default=1,
        type=int,
        help="first image id in generated COCO json",
    )
    parser.add_argument(
        "--start-annotation-id",
        default=1,
        type=int,
        help="first annotation id in generated COCO json",
    )
    return parser


def read_classes(classes_file):
    classes = []
    with classes_file.open("r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                classes.append(name)
    if not classes:
        raise ValueError(f"No class names found in {classes_file}")
    return classes


def iter_images(images_dir):
    return sorted(
        path for path in images_dir.rglob("*") if path.suffix.lower() in IMG_SUFFIXES
    )


def yolo_box_to_coco(parts, img_w, img_h, label_file, line_no):
    if len(parts) != 5:
        raise ValueError(f"{label_file}:{line_no} should contain 5 values")

    class_id = int(float(parts[0]))
    x_center, y_center, box_w, box_h = [float(v) for v in parts[1:]]

    x_min = (x_center - box_w / 2.0) * img_w
    y_min = (y_center - box_h / 2.0) * img_h
    width = box_w * img_w
    height = box_h * img_h

    x_min = max(0.0, min(x_min, img_w - 1.0))
    y_min = max(0.0, min(y_min, img_h - 1.0))
    width = max(0.0, min(width, img_w - x_min))
    height = max(0.0, min(height, img_h - y_min))
    return class_id, [x_min, y_min, width, height]


def convert(args):
    if not args.images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")
    if not args.labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {args.labels_dir}")

    classes = read_classes(args.classes)
    categories = [
        {"id": idx, "name": name, "supercategory": "object"}
        for idx, name in enumerate(classes)
    ]

    images = []
    annotations = []
    image_id = args.start_image_id
    ann_id = args.start_annotation_id

    for image_path in iter_images(args.images_dir):
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
        img_h, img_w = img.shape[:2]

        rel_image = image_path.relative_to(args.images_dir).as_posix()
        images.append(
            {
                "id": image_id,
                "file_name": rel_image,
                "width": img_w,
                "height": img_h,
            }
        )

        label_file = args.labels_dir / image_path.relative_to(args.images_dir)
        label_file = label_file.with_suffix(".txt")
        if label_file.exists():
            with label_file.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    class_id, bbox = yolo_box_to_coco(
                        line.split(), img_w, img_h, label_file, line_no
                    )
                    if not 0 <= class_id < len(classes):
                        raise ValueError(
                            f"{label_file}:{line_no} class id {class_id} is out of range"
                        )
                    area = bbox[2] * bbox[3]
                    if area <= 0:
                        continue
                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": class_id,
                            "bbox": bbox,
                            "area": area,
                            "iscrowd": 0,
                            "segmentation": [],
                        }
                    )
                    ann_id += 1

        image_id += 1

    coco = {
        "info": {"description": "Converted from YOLO format"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(
        f"Saved {args.output}: {len(images)} images, "
        f"{len(annotations)} annotations, {len(categories)} classes"
    )


if __name__ == "__main__":
    convert(make_parser().parse_args())
