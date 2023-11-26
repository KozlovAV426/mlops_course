from dataclasses import dataclass


@dataclass
class PathParams:
    model: str
    optimizer: str
    metrics: str
    dataset: str


@dataclass
class ModelParams:
    batch_size: int
    n_epoch: int


@dataclass
class Params:
    paths: PathParams
    model: ModelParams
    mlflow_url: str
