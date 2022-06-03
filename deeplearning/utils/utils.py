from typing import Dict

import xml.etree.ElementTree as ET
import numpy as np

from utils.types import Annotation, BoundingBox

def parse_annotation(annotation_path : str, classes : Dict[str, int], return_biggest : bool = False) -> Annotation:
    root = ET.parse(annotation_path)

    size = root.find('size')
    width = float(size.find("width").text)
    height = float(size.find("height").text)

    annotation = Annotation(
        width=width,
        height=height,
        labels=[],
        boxes=[],
        areas=[]
    )
    
    for child in root.findall('object'):
        label = child.find('name').text
        
        bndbox = child.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        xmax = float(bndbox.find('xmax').text)
        ymin = float(bndbox.find('ymin').text)
        ymax = float(bndbox.find('ymax').text)

        bbox = BoundingBox(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax
        )

        area = (bbox.ymax - bbox.ymin) * (bbox.xmax - bbox.xmin)

        annotation['labels'].append(classes[label])
        annotation['boxes'].append(list(bbox))
        annotation['areas'].append(area)

    if return_biggest:
        idx = np.argmax(annotation['areas'])
        label = annotation['labels'][idx]
        bbox = annotation['boxes'][idx]
        area = annotation['areas'][idx]
        annotation = Annotation(
            width=width,
            height=height,
            labels=[label],
            boxes=[bbox],
            areas=[area]
        )

    return annotation