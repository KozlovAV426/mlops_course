import os
from typing import Any

import git
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn


PATH_TO_METRICS = "metrics.csv"
PATH_TO_OPTIMIZER = ".results/optimizer.pth"
PATH_TO_MODEL = ".results/model.pth"


class TrainerModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = None,
        weight_decay: float = None,
        optim: torch.optim.Optimizer = None,
        criterion: nn.Module = None,
    ):
        super().__init__()

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.optim = optim

        self.criterion = criterion or nn.CrossEntropyLoss()

    def forward(self, *args: Any, **kwargs: Any):
        return self.model.forward(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = self.optim or torch.optim.SGD(
            self.model.parameters(), lr=self.lr or 0.01, momentum=self.momentum or 0.5
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        self.model.train(True)

        logits = self.model(x_batch)
        loss = self.criterion(logits, y_batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True)

        self.log(
            "train_accuracy",
            (logits.detach().cpu().argmax(axis=1) == y_batch).float().mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        self.model.train(False)

        with torch.no_grad():
            logits = self.model(x_batch)
            loss = self.criterion(logits, y_batch)
            self.log("val_loss", loss, on_step=True, on_epoch=True)

            self.log(
                "val_accuracy",
                (logits.detach().cpu().argmax(axis=1) == y_batch).float().mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

            return loss

    def predict_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        self.model.train(False)

        with torch.no_grad():
            logits = self.model(x_batch)
            predictions = logits.detach().cpu().argmax(axis=1)

            return {
                "true_label": y_batch,
                "predicted_label": predictions,
                "logits": logits,
                "loss": self.criterion(logits, y_batch).item(),
            }


class MnistSolver:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer = None,
        lr: float = None,
        momentum: float = None,
        criterion: nn.Module = None,
        path_to_model: str = None,
        path_to_optimizer: str = None,
        path_to_metrics: str = None,
        n_epoch: int = None,
        mflow_url: str = None,
    ):
        self.model = model
        self.n_epoch = n_epoch or 1

        self.optim = optimizer or torch.optim.SGD(
            model.parameters(), lr=lr or 0.01, momentum=momentum or 0.5
        )
        self.criterion = criterion or nn.CrossEntropyLoss()

        self.path_to_model = path_to_model or PATH_TO_MODEL
        self.path_to_optimizer = path_to_optimizer or PATH_TO_OPTIMIZER
        self.path_to_metrics = path_to_metrics or PATH_TO_METRICS
        self.mlflow_url = mflow_url

        os.makedirs(self.path_to_model.split('/')[0], exist_ok=True)
        os.makedirs(self.path_to_optimizer.split('/')[0], exist_ok=True)

        self.trainer = TrainerModule(
            model=model,
            lr=lr,
            optim=self.optim,
            criterion=criterion,
        )

        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha

        mlflow_logger = pl.loggers.MLFlowLogger(
            experiment_name='mnist', tracking_uri=self.mlflow_url
        )
        mlflow_logger.log_hyperparams(
            {'git_commit_id': sha, 'n_epoch': n_epoch, 'lr': lr, 'momentum': momentum}
        )

        self.pl_trainer = pl.Trainer(
            precision="bf16-mixed",
            max_epochs=self.n_epoch,
            log_every_n_steps=1,
            logger=[
                mlflow_logger,
                pl.loggers.CSVLogger("logs"),
            ],
        )

    def fit(self, data_module: pl.LightningDataModule):
        self.pl_trainer.fit(self.trainer, data_module)

    def predict(self, dataloader, path=None):
        prediction = self.pl_trainer.predict(
            model=self.trainer, dataloaders=[dataloader]
        )[0]

        true_labels = prediction["true_label"]
        predicted = prediction["predicted_label"]

        metrics = {
            "avg_loss": prediction["loss"],
            "accuracy": [(true_labels.numpy() == predicted.numpy()).mean()],
        }
        df = pd.DataFrame(metrics)

        if path:
            df.reset_index().round(3).to_csv(path, index=False)

    def save_model(self, path=None):
        self.pl_trainer.save_checkpoint(path or self.path_to_model, weights_only=True)

    def load_model(self, path=None):
        self.trainer = TrainerModule.load_from_checkpoint(
            path or self.path_to_model, model=self.model
        )
