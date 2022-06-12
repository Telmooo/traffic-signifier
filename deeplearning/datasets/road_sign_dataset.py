from typing import Dict, List

import os
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
import numpy as np
import cv2

from utils.utils import parse_annotation


class RoadSignDataset(Dataset):
    img_names: List[str]
    img_dir: str
    annotation_dir: str
    is_train: bool
    multilabel: bool
    obj_detection: bool
    classes: Dict[str, int]
    def __init__(self, img_names: List[str], img_dir: str, annotation_dir: str, classes: Dict[str, int], is_train: bool = True, multilabel: bool = True, obj_detection : bool = False) -> None:
        self.img_names = img_names
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.classes = classes
        self.is_train = is_train
        self.multilabel = multilabel
        self.obj_detection = obj_detection

        self.transform = None
        if is_train:
            self.transform = A.Compose([
                A.OneOf([ # COLOR AUGMENTATION
                    A.Posterize(num_bits=4, p=0.3),
                    A.CLAHE(clip_limit=2, tile_grid_size=(8,8), p=0.25)
                ], p=1.0),
                # A.OneOf([ # LIGHTING AUGMENTATION
                A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=True, p=0.25),
                    # A.RandomGamma(p=1.0, gamma_limit=(40, 90)) # Implementation on current version is broken
                # ], p=1.0),
                # A.OneOf([ # WEATHER AUGMENTATION
                #     A.RandomSunFlare(flare_roi=(0,0,1,0.5), angle_lower=0, angle_upper=1,
                #                     num_flare_circles_lower=3, num_flare_circles_upper=5,
                #                     src_radius=50, src_color=(255,255,255), p=0.05),
                #     A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.4, alpha_coef=0.08, p=0.15)
                # ], p=1.0),
                A.HorizontalFlip(p=0.4),
                A.OneOf([ # GEOMETRIC AUGMENTATION
                    A.Rotate(limit=(-30, 30), interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101,
                                value=None, mask_value=None, p=0.25),
                    A.Perspective(scale=(0.05,0.2), keep_size=True, pad_mode=cv2.BORDER_CONSTANT, pad_val=0, mask_pad_val=0,
                                fit_output=False, interpolation=cv2.INTER_LINEAR, p=0.25)
                ], p=1.0),
                A.OneOf([ # BLUR AUGMENTATION
                    A.Blur(blur_limit=(1, 3), p=0.1),
                    # A.AdvancedBlur(blur_limit=(3, 7), sigmaX_limit=(0.2, 1.0),
                    #                 sigmaY_limit=(0.2, 1.0), rotate_limit=90,
                    #                 beta_limit=(0.5, 8.0), noise_limit=(0.9, 1.1), p=1.0),
                    A.MotionBlur(blur_limit=(3, 7), p=0.2)
                ], p=1.0),
                # A.OneOf([ # DISTORTION AUGMENTATION
                #     A.Downscale(scale_min=0.80, scale_max=0.9, p=0.35),
                #     A.GaussNoise(var_limit=(10.0, 75.0), mean=0, per_channel=True, p=0.2)
                # ], p=1.0),
                A.ToGray(p=0.05),
                # # A.OneOf([ # SCALING AUGMENTATION
                # # ], p=1.0),
                A.PadIfNeeded(min_height=224, min_width=224, p=1.0),
                A.RandomCrop(height=224, width=224, p=1.0),
                # ImageNet normalization since DenseNet was trained with that
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], p=1.0, bbox_params=A.BboxParams(format="pascal_voc", min_area=0, min_visibility=0, label_fields=["class_labels", "areas"]))

        else:
            self.transform = A.Compose([
                # ImageNet normalization since DenseNet was trained with that
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], p=1.0, bbox_params=A.BboxParams(format="pascal_voc", min_area=0, min_visibility=0, label_fields=["class_labels", "areas"]))

    def __len__(self) -> int:
        return len(self.img_names)
    
    def __getitem__(self, index: int):
        img_name = self.img_names[index]
        img_id = int(img_name.split("road")[1])
        img_path = os.path.join(self.img_dir, f"{img_name}.png")

        image = np.array(Image.open(img_path).convert("RGB"))

        annotation = parse_annotation(
            annotation_path=os.path.join(self.annotation_dir, f"{img_name}.xml"),
            classes=self.classes,
            return_biggest=(not self.multilabel and not self.obj_detection )
        )

        img_width, img_height = annotation["width"], annotation["height"]

        transformed = self.transform(
            image=image,
            bboxes=annotation["boxes"],
            class_labels=annotation["labels"],
            areas=annotation["areas"],
        )

        transformed_img = transformed["image"]
        transformed_boxes = transformed["bboxes"]
        transformed_labels = transformed["class_labels"]
        transformed_areas = transformed["areas"]

        if self.multilabel:
            labels = [0] * len(self.classes)
            for label in transformed_labels:
                labels[label] = 1
            
            transformed_labels = labels

            transformed_boxes = [[0, 0, img_width, img_height]]
            transformed_areas = [img_width * img_height]

        if not transformed_labels:
            transformed_labels = [self.classes["background"]]
            transformed_boxes = [[0, 0, img_width, img_height]]
            transformed_areas = [img_width * img_height]
        
        target = {
            "labels": torch.as_tensor(data=transformed_labels, dtype=torch.int64),
            "boxes": torch.as_tensor(transformed_boxes, dtype=torch.float32),
            "areas": torch.as_tensor(transformed_areas, dtype=torch.float32),
            "imageId": torch.as_tensor(img_id, dtype=torch.int16)
        }

        return transformed_img.float(), target

    def collate_fn(self, batch):
        if self.obj_detection:
            return tuple(zip(*batch))

        if self.multilabel:
            images = list()
            labels = list()
            imageIds = list()

            for item in batch:
                images.append(item[0])
                target = item[1]
                labels.append(target["labels"])        
                imageIds.append(target["imageId"])

            
            images = torch.stack(images, dim=0)
            labels = torch.stack(labels, dim=0)
            imageIds = torch.stack(imageIds, dim=0)

            targets = {
                "labels": labels,
                "imageIds": imageIds
            }
            return images, targets

        images = list()
        labels = list()
        boxes = list()
        areas = list()
        imageIds = list()

        for item in batch:
            images.append(item[0])
            target = item[1]
            labels.append(target["labels"])
            boxes.append(target["boxes"])
            areas.append(target["areas"])
            imageIds.append(target["imageId"])
        
        images = torch.stack(images, dim=0)
        labels = torch.stack(labels, dim=1)[0]
        boxes = torch.stack(boxes, dim=1)[0]
        areas = torch.stack(areas, dim=1)[0]
        imageIds = torch.stack(imageIds, dim=0)

        targets = {
            "labels": labels,
            "boxes": boxes,
            "areas": areas,
            "imageIds": imageIds
        }

        return images, targets