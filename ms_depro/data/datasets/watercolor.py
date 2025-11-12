import os
import os.path as osp
import xml.etree.ElementTree as ET
from pymage_size import get_image_size
from tqdm import tqdm

from detectron2.structures import BoxMode

from ms_depro.data import DatasetCatalog, MetadataCatalog


ALL_CATEGORIES = [
    "bicycle", "bird", "car", "cat", "dog", "person"
    ]


def get_annotation(root, image_id, ind, depth):
    annotation_file = osp.join(root, "Annotations", "%s.xml" % image_id)

    et = ET.parse(annotation_file)
    
    objects = et.findall("object")                                              

    record = {}
    record["file_name"] = osp.join(root,  "JPEGImages", "%s.jpg" % image_id)
    record["file_name_depth"] = osp.join(root,  depth+"Images", "%s.jpg" % image_id)
    img_format = get_image_size(record["file_name"])
    w, h = img_format.get_dimensions()

    record["image_id"] = image_id
    record["annotations"] = []

    for obj in objects:
        class_name = obj.find('name').text.lower().strip()
        if class_name not in ALL_CATEGORIES:
            continue
        
        if obj.find('pose') is None:
            obj.append(ET.Element('pose'))
            obj.find('pose').text = '0'

        if obj.find('truncated') is None:
            obj.append(ET.Element('truncated'))
            obj.find('truncated').text = '0'

        if obj.find('difficult') is None:
            obj.append(ET.Element('difficult'))
            obj.find('difficult').text = '0'

        bbox = obj.find('bndbox')
        # VOC dataset format follows Matlab, in which indexes start from 0
        x1 = max(0,float(bbox.find('xmin').text) - 1) # fixing when -1 in anno
        y1 = max(0,float(bbox.find('ymin').text) - 1) # fixing when -1 in anno
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1
        box = [x1, y1, x2, y2]
        
        bbox.find('xmin').text = str(int(x1))
        bbox.find('ymin').text = str(int(y1))
        bbox.find('xmax').text = str(int(x2))
        bbox.find('ymax').text = str(int(y2))

        record_obj = {
            "bbox": box,
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": ALL_CATEGORIES.index(class_name),
            }
        record["annotations"].append(record_obj)

    if len(record["annotations"]):
        # to convert float to int
        # et.write(annotation_file)
        record["height"] = h
        record["width"] = w
        return record

    else:
        return None

def files2dict(root, split, depth):
    print(split)
    dataset_dicts = []
    image_sets_file = osp.join(root, "ImageSets", "Main", "%s.txt" % split)

    with open(image_sets_file) as f:
        count = 0
        for line in tqdm(f):
            record = get_annotation(root, line.rstrip(), count, depth)
            if record is not None:
                dataset_dicts.append(record)
                count +=1
                
    return dataset_dicts

def register_watercolor(dataset_root, depth):
    dataset_list = ['watercolor']
    settype = ['train','test']
    
    for name in dataset_list:
        dir_name = name.capitalize()
        for ind, d in enumerate(settype):
            DatasetCatalog.register(name+"_" + d, lambda root=dataset_root, name=name, d=d \
                : files2dict(osp.join(dataset_root, dir_name), d, depth))
            MetadataCatalog.get(name+ "_" + d).set(thing_classes=ALL_CATEGORIES, evaluator_type='coco')
            MetadataCatalog.get(name+ "_" + d).set(dirname=osp.join(dataset_root, name))
            MetadataCatalog.get(name+ "_" + d).set(split=d)
            MetadataCatalog.get(name+ "_" + d).set(year=2007)