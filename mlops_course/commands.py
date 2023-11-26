import fire
import hydra
from hydra.core.config_store import ConfigStore

from mlops_course.config import Params
from mlops_course.data import DataModule
from mlops_course.model.model import Net
from mlops_course.trainer.mnist_solver import MnistSolver


cs = ConfigStore.instance()
cs.store(name="params", node=Params)


class Starter:
    def __init__(self):
        with hydra.initialize(config_path="../conf", version_base="1.3"):
            self.params: Params = hydra.compose(config_name="config")

            self.datamodule = DataModule(
                dataset_path=self.params.paths.dataset,
                train_ratio=0.1,
                batch_size=self.params.model.batch_size,
            )
            self.model = Net()
            self.trainer = MnistSolver(
                model=self.model,
                n_epoch=self.params.model.n_epoch,
                mflow_url=self.params.mlflow_url,
            )

    def train(self):
        self.trainer.fit(self.datamodule)
        self.trainer.save_model(self.params.paths.model)

    def infer(self):
        self.trainer.load_model(self.params.paths.model)
        self.trainer.predict(
            self.datamodule.test_dataloader(), path=self.params.paths.metrics
        )


def main():
    fire.Fire(Starter())


if __name__ == "__main__":
    main()
