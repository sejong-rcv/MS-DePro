# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import os.path as osp
from detectron2.config import get_cfg
from ms_depro import add_cfg

from .diverse_weather import register_diverse_weather
from .pascal_voc import register_pascal_voc
from .coco import register_coco
from .clipart import register_clipart
from .comic import register_comic
from .watercolor import register_watercolor
from .bdd100k import register_bdd100k, register_caronly_bdd100k
from .cityscapes import register_cityscapes, register_caronly_cityscapes
from .kitti import register_caronly_kitti
from .coco_md import register_md_coco
from .synscapes import register_synscapes
 
# register dataset
_root = "datasets"

cfg = get_cfg()
add_cfg(cfg)
depth_map = cfg.DATALOADER.DEPTH

register_diverse_weather(_root, depth_map)
register_pascal_voc(_root, depth_map)
register_coco(_root, depth_map)
register_clipart(_root, depth_map)
register_comic(_root, depth_map)
register_watercolor(_root, depth_map)
register_bdd100k(_root, depth_map)
register_caronly_bdd100k(_root, depth_map)
register_caronly_cityscapes(_root, depth_map)
register_caronly_kitti(_root, depth_map)
register_cityscapes(_root, depth_map)
register_md_coco(_root, depth_map)
register_synscapes(_root, depth_map)