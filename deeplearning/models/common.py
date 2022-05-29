from typing import List, TypedDict


class Hyperparameters(TypedDict, total=False):
    learning_rate: float
    momentum: float


class ScoreHistory(TypedDict):
    loss: List[float]
    metric: List[float]
