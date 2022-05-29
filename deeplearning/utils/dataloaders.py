from typing import List, NamedTuple, TypedDict
from torch.utils.data import Dataset

import os

import xml.etree.ElementTree as ET
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
from PIL import Image

class BoundingBox(NamedTuple):
    xmin: float
    ymin: float
    xmax: float
    ymax: float

class Annotation(TypedDict):
    labels: List[int]
    bboxes: List[List[float]]
    areas: List[float]

def parse_annotation(annotation_path, classes, return_biggest : bool = False) -> Annotation:
    root = ET.parse(annotation_path)
    
    annotation = Annotation(
        labels=[],
        bboxes=[],
        areas=[]
    )
    
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

        area = (bbox.ymax - bbox.ymin) * (bbox.xmax - bbox.xmin)

        annotation['labels'].append(classes[label])
        annotation['bboxes'].append(list(bbox))
        annotation['areas'].append(area)

    if return_biggest:
        idx = np.argmax(annotation['areas'])
        label = annotation['labels'][idx]
        bbox = annotation['bboxes'][idx]
        area = annotation['areas'][idx]
        annotation = Annotation(
            labels=[label],
            bboxes=[bbox],
            areas=[area]
        )

    return annotation

class RoadSignDataset(Dataset):
    def __init__(self, images_filenames, images_directory, annotations_directory, is_train : bool = True, multilabel : bool = False):
        self.images_filenames = images_filenames
        self.img_dir = images_directory
        self.annotations_dir = annotations_directory
        self.transform = None
        self.multilabel = multilabel

        self.classes = {
            "trafficlight": 0,
            "speedlimit": 1,
            "crosswalk": 2,
            "stop": 3
        }
        if is_train:
            self.transform = A.Compose([
                A.OneOf([ # COLOR AUGMENTATION
                    A.Posterize(num_bits=4, p=0.4),
                    A.CLAHE(clip_limit=2, tile_grid_size=(8,8), p=0.3)
                ], p=1.0),
                # A.OneOf([ # LIGHTING AUGMENTATION
                A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=True, p=0.35),
                    # A.RandomGamma(p=1.0, gamma_limit=(40, 90)) # Implementation on current version is broken
                # ], p=1.0),
                A.OneOf([ # WEATHER AUGMENTATION
                    A.RandomSunFlare(flare_roi=(0,0,1,0.5), angle_lower=0, angle_upper=1, 
                                    num_flare_circles_lower=3, num_flare_circles_upper=5, 
                                    src_radius=50, src_color=(255,255,255), p=0.05),
                    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.4, alpha_coef=0.08, p=0.15)
                ], p=1.0),
                A.HorizontalFlip(p=1.0),
                A.OneOf([ # GEOMETRIC AUGMENTATION
                    A.Rotate(limit=(-30, 30), interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, 
                                value=None, mask_value=None, p=0.3),
                    A.Perspective(scale=(0.05,0.2), keep_size=True, pad_mode=cv2.BORDER_CONSTANT, pad_val=0, mask_pad_val=0, 
                                fit_output=False, interpolation=cv2.INTER_LINEAR, p=0.5)
                ], p=1.0),
                A.OneOf([ # BLUR AUGMENTATION
                    A.Blur(blur_limit=(1, 3), p=0.1),
                    # A.AdvancedBlur(blur_limit=(3, 7), sigmaX_limit=(0.2, 1.0),
                    #                 sigmaY_limit=(0.2, 1.0), rotate_limit=90,
                    #                 beta_limit=(0.5, 8.0), noise_limit=(0.9, 1.1), p=1.0),
                    A.MotionBlur(blur_limit=(3, 7), p=0.2)
                ], p=1.0),
                A.OneOf([ # DISTORTION AUGMENTATION
                    A.Downscale(scale_min=0.80, scale_max=0.9, p=0.35),
                    A.GaussNoise(var_limit=(10.0, 75.0), mean=0, per_channel=True, p=0.2)
                ], p=1.0),
                A.ToGray(p=0.05),
                A.OneOf([ # SCALING AUGMENTATION
                ], p=1.0),
                A.PadIfNeeded(min_height=224, min_width=224, p=1.0),
                A.RandomCrop(height=224, width=224, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet normalization since DenseNet was trained with that
                ToTensorV2()
            ], p=1.0, bbox_params=A.BboxParams(format="albumentations", min_area=0, min_visibility=0, label_fields=["class_labels", "areas"]))

        else:
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet normalization since DenseNet was trained with that
                ToTensorV2()
            ], p=1.0, bbox_params=A.BboxParams(format="albumentations", min_area=0, min_visibility=0, label_fields=["class_labels", "areas"]))


    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, index: int):
        image_filename = self.images_filenames[index]
        img_path = os.path.join(self.img_dir, f"{image_filename}.png")

        image = np.array(Image.open(img_path).convert("RGB"))

        annotation = parse_annotation(os.path.join(self.annotations_dir, f"{image_filename}.xml"), classes=self.classes, return_biggest=(not self.multilabel))

        if self.transform is not None:
            transformed = self.transform(image=image, bboxes=annotation['bboxes'], class_labels=annotation['labels'], areas=annotation['areas'])
            transformed_image = transformed["image"]
            transformed_bboxes = transformed["bboxes"]
            transformed_class_labels = transformed["class_labels"]
            transformed_areas = transformed["areas"]

            if not transformed_class_labels:
                transformed_class_labels = [-1]
                transformed_bboxes = [[-1, -1, -1, -1]]
                transformed_areas = [-1]

            target = {
                "labels": torch.as_tensor(transformed_class_labels, dtype=torch.int64),
                "bboxes": torch.as_tensor(transformed_bboxes, dtype=torch.float32),
                "areas": torch.as_tensor(transformed_areas, dtype=torch.float32)
            }

        return transformed_image.float(), target

    def collate_fn(self, batch):
        images = list()
        labels = list()
        bboxes = list()
        areas = list()


        for b in batch:
            images.append(b[0])
            target = b[1]
            labels.append(target["labels"])
            bboxes.append(target["bboxes"])
            areas.append(target["areas"])

        images = torch.stack(images, dim=0)
        if not self.multilabel:
            labels = torch.stack(labels, dim=1)[0]
            bboxes = torch.stack(bboxes, dim=1)[0]
            areas = torch.stack(areas, dim=1)[0]

        target = {
            "labels": labels,
            "bboxes": bboxes,
            "areas": areas,
        }
        
        return images, target


