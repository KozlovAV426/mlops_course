from dataclasses import dataclass


@dataclass
class PathParams:
    model: str
    optimizer: str
    metrics: str


@dataclass
class ModelParams:
    train_batch_size: int
    test_batch_size: int
    n_epoch: int


@dataclass
class Params:
    paths: PathParams
    model: ModelParams
