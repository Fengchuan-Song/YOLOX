#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import copy

from train import build_coco_annotations, configure_module, make_parser, train_one_model


if __name__ == "__main__":
    configure_module()
    args = make_parser().parse_args()
    args.models = ["nano"]

    ann_files = build_coco_annotations(
        dataset_root=args.dataset_root,
        splits=("train", "val"),
        classes=args.classes,
        ann_cache_dir=args.ann_cache_dir,
        force=args.force_rebuild_ann,
    )

    train_one_model(copy.deepcopy(args), "nano", ann_files)
