import os.path as osp
import numpy as np
import functools
import multiprocessing as mp
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union

from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from detectron2.utils.comm import get_world_size

from ms_depro.data import DatasetCatalog, MetadataCatalog


ALL_CATEGORIES = [
    'car' # car-only
    ]



def load_kitti_instances(dirname: str, split: str, depth: str, class_names: Union[List[str], Tuple[str, ...]]):
    """
    Load KITTI detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        depth (str): "depth_anything", "depth_pro" or others
        class_names: list or tuple of class names
    """
    with PathManager.open(osp.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileid = np.loadtxt(f, dtype=str)

    # Needs to read many small annotation files. Makes sense at local
    annotation_dirname = PathManager.get_local_path(osp.join(dirname, "Annotations/"))
    pool = mp.Pool(processes=max(mp.cpu_count() // get_world_size() // 2, 4))
    dataset_dicts = pool.map(
                    functools.partial(create_dict,
                                      annotation_dirname=annotation_dirname, 
                                      dirname=dirname, 
                                      split=split, 
                                      depth=depth, 
                                      class_names=class_names),
                    fileid
                    )
    pool.close()
    return dataset_dicts

def create_dict(fileid, annotation_dirname, dirname, split, depth, class_names):
    anno_file = osp.join(annotation_dirname, fileid + ".xml")
    image_file = osp.join(dirname, "JPEGImages", fileid + ".jpg")
    depth_file = osp.join(dirname, depth + "Images", fileid + ".jpg")
        
    with PathManager.open(anno_file) as f:
        tree = ET.parse(f)

    r = {
        "file_name": image_file,
        "file_name_depth": depth_file,
        "image_id": fileid,
        "height": int(tree.findall("./size/height")[0].text),
        "width": int(tree.findall("./size/width")[0].text),
        }
    
    instances = []
    for obj in tree.findall("object"):
        cat = obj.find("name").text
        bbox = obj.find("bndbox")
        bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]

        bbox[0] -= 1.0
        bbox[1] -= 1.0
        if cat in class_names:
            instances.append(
                {"category_id": class_names.index(cat), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
    r["annotations"] = instances
    return r


def register_kitti_instances(name, dirname, split, depth, year=2012, class_names=ALL_CATEGORIES):
    DatasetCatalog.register(name, lambda: load_kitti_instances(dirname, split, depth, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )
   
def register_caronly_kitti(dataset_root, depth):
    SPLITS = [
        ("kitti_caronly_trainval", "Kitti", "trainval"),
        ("kitti_caronly_train", "Kitti", "train"),
        ("kitti_caronly_val", "Kitti", "val")
    ]
    for name, dirname, settype in SPLITS:
        register_kitti_instances(name, osp.join(dataset_root, dirname), settype, depth)
        MetadataCatalog.get(name).evaluator_type = "coco"