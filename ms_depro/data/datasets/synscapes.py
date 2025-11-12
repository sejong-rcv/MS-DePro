import os
import os.path as osp
import json
import logging
import functools
import multiprocessing as mp
from PIL import Image

from detectron2.structures import Boxes
from detectron2.utils.comm import get_world_size

from ms_depro.data import DatasetCatalog, MetadataCatalog


logger = logging.getLogger(__name__)

def create_dict(images, json_path, image_path, depth, mapper):
    with open(f'{json_path}/{images}.json') as f:
        x = json.load(f)
    data_dict = {}
    data_dict["file_name"] = f'{image_path}/{images}.png'
    data_dict["file_name_depth"] = f'{image_path.replace("img/rgb-2k", depth+"Images")}/{images}.png'
    
    image_size = Image.open(data_dict["file_name_depth"]).size # synscapes/img/rgb-2k
    data_dict["height"] = image_size[1]
    data_dict["width"] = image_size[0]
    data_dict["image_id"] = images

    boxes = x['instance']['bbox2d'].keys()
    annos = []
    for item in boxes:
        if (x['instance']['class'][item] in mapper.keys()):
            anno = {}
            anno["bbox"] = [
                (x['instance']['bbox2d'][item]['xmin'] * x['camera']['intrinsic']['resx']) * (data_dict["width"] / x['camera']['intrinsic']['resx']),
                (x['instance']['bbox2d'][item]['ymin'] * x['camera']['intrinsic']['resy']) * (data_dict["height"] / x['camera']['intrinsic']['resy']),
                (x['instance']['bbox2d'][item]['xmax'] * x['camera']['intrinsic']['resx']) * (data_dict["width"] / x['camera']['intrinsic']['resx']),
                (x['instance']['bbox2d'][item]['ymax'] * x['camera']['intrinsic']['resy']) * (data_dict["height"] / x['camera']['intrinsic']['resy'])
            ]
            anno["bbox_mode"] = 0
            anno["category_id"] = mapper[x['instance']['class'][item]]
            annos.append(anno)
    data_dict['annotations'] = annos
    return data_dict

def get_synscapes_dataset(name, json_file, image_root, depth, thing_classes):
    meta = MetadataCatalog.get(name)
    meta.thing_classes = thing_classes
    logger.info("Loading Synscapes dataset")
    
    from cityscapesscripts.helpers.labels import labels

    labels = [l for l in labels if l.hasInstances and not l.ignoreInEval and l.name in thing_classes]
    dataset_id_to_contiguous_id = {l.id: idx for idx, l in enumerate(labels)}
    meta.thing_dataset_id_to_contiguous_id = dataset_id_to_contiguous_id

    images = list(range(1, 25001))
    pool = mp.Pool(processes=max(mp.cpu_count() // get_world_size() // 2, 4))
    dataset_dicts = pool.map(
                    functools.partial(create_dict, json_path=json_file, image_path=image_root, depth=depth, mapper=dataset_id_to_contiguous_id),
                    images,   
                    )
    pool.close()
    return dataset_dicts

def register_synscapes_instances(name, metadata, json_file, image_root, depth, thing_classes):
    """
    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        depth (str): "depth_anything", "depth_pro" or others.
        image_root (str or path-like): directory which contains all the images.
        thing_classes (list): list of categories
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: get_synscapes_dataset(name, json_file, image_root, depth, thing_classes))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, depth=depth, evaluator_type="coco", **metadata
    )

def register_synscapes(dataset_root, depth):
    metadata={'thing_classes':["person", "car", "rider", "truck", "motorcycle", "bicycle", "bus"]}
    SPLITS = (
        ["synscapes_train", "Synscapes/meta", "Synscapes/img/rgb-2k"],
        ["synscapes_val", "Synscapes/meta", "Synscapes/img/rgb-2k"]
    )
    for name, json_file, image_root in SPLITS:
        register_synscapes_instances(name=name, 
                                     metadata=metadata, 
                                     json_file=osp.join(dataset_root, json_file), 
                                     image_root=osp.join(dataset_root, image_root), 
                                     depth=depth,
                                     thing_classes=metadata['thing_classes'])