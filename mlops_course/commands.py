import subprocess

import fire
import hydra
import mlflow
import onnx
import torch
from hydra.core.config_store import ConfigStore
from mlflow.models import infer_signature

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

    def run_server(self):
        onnx_model = onnx.load_model("mnist_solver.onnx")
        with mlflow.start_run() as _:
            X = torch.randn(1, 1, 28, 28)
            signature = infer_signature(X.numpy(), self.model(X).detach().numpy())
            mlflow.onnx.save_model(onnx_model, "mnist_solver_model", signature=signature)

        # starting mlflow model serving
        subprocess.run(
            [
                "poetry",
                "run",
                "mlflow",
                "models",
                "serve",
                "-p",
                "5001",
                "-m",
                "mnist_solver_model",
                "--env-manager=local",
            ]
        )

    def train(self):
        self.trainer.fit(self.datamodule)
        self.trainer.save_model(self.params.paths.model)
        torch.onnx.export(
            self.model,
            torch.randn(1, 1, 28, 28),
            "mnist_solver.onnx",
            export_params=True,
            opset_version=10,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        )

    def infer(self):
        self.trainer.load_model(self.params.paths.model)
        self.trainer.predict(
            self.datamodule.test_dataloader(), path=self.params.paths.metrics
        )


def main():
    fire.Fire(Starter())


if __name__ == "__main__":
    main()
