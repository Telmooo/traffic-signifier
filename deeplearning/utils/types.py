from typing import List, NamedTuple, TypedDict

from torch import Tensor

class BoundingBox(NamedTuple):
    xmin: float
    ymin: float
    xmax: float
    ymax: float

class Annotation(TypedDict):
    width: float
    height: float
    labels: List[int]
    boxes: List[List[float]]
    areas: List[float]
