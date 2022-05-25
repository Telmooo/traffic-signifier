"""
IMAGENET Normalization
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

"""
from typing import List, NamedTuple, TypedDict
from torch.utils.data import Dataset
import os

import xml.etree.ElementTree as ET
from torchvision.io import read_image
import albumentations as A 

class BoundingBox(NamedTuple):
    xmin: float
    ymin: float
    xmax: float
    ymax: float

class Annotation(TypedDict):
    labels: List[str]
    bbox: List[BoundingBox]

def parse_annotation(annotation_path):
    root = ET.parse(annotation_path)
    
    annotation = Annotation()
    
    filename, _extension = os.path.splitext(os.path.basename(filename))

    size = root.find('size')
    width = float(size.find("width").text)
    height = float(size.find("height").text)
    
    for child in root.findall('object'):
        label = child.find('name').text
        
        bndbox = child.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        xmax = float(bndbox.find('xmax').text)
        ymin = float(bndbox.find('ymin').text)
        ymax = float(bndbox.find('ymax').text)

        bbox = BoundingBox(
            xmin=xmin / width,
            ymin=ymin / height,
            xmax=xmax / width,
            ymax=ymax / height
        )

        annotation['labels'].append(label)
        annotation['bbox'].append(bbox)

    return annotation

class RoadSignDataset(Dataset):
    def __init__(self, images_filenames, images_directory, annotations_directory, use_transform : bool = True):
        self.images_filenames = images_filenames
        self.img_dir = images_directory
        self.annotations_dir = annotations_directory
        self.transform = None
        if use_transform:
            self.transform = A.Compose([
                
            ], bbox_params=A.BboxParams(format="albumentations", min_area=0, min_visibility=0, label_fields=["class_labels"]))

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, index: int):
        image_filename = self.images_filenames[index]
        img_path = os.path.join(self.img_dir, f"{image_filename}.png")
        image = read_image(img_path)
        annotation = parse_annotation(os.path.join(self.annotations_dir, f"{image_filename}.xml"))

        if self.transform is not None:
            transformed_image = self.transform(image=image)
            image = transformed_image

        
        return super().__getitem__(index)



        