import os

import pandas as pd
import torch
import torch.optim as optim
from torch import nn


PATH_TO_METRICS = "metrics.csv"
PATH_TO_OPTIMIZER = ".results/optimizer.pth"
PATH_TO_MODEL = ".results/model.pth"


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
    ):
        self.model = model

        self.optim = optimizer or optim.SGD(
            model.parameters(), lr=lr or 0.01, momentum=momentum or 0.5
        )
        self.criterion = criterion or nn.CrossEntropyLoss()

        self.path_to_model = path_to_model or PATH_TO_MODEL
        self.path_to_optimizer = path_to_optimizer or PATH_TO_OPTIMIZER
        self.path_to_metrics = path_to_metrics or PATH_TO_METRICS

        os.makedirs(self.path_to_model.split('/')[0], exist_ok=True)
        os.makedirs(self.path_to_optimizer.split('/')[0], exist_ok=True)

    def train_epoch(self, epoch, train_loader, log_interval=10):
        train_losses = []
        train_counter = []

        for batch_idx, (data, target) in enumerate(train_loader):
            self.optim.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optim.step()

            if batch_idx % log_interval == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset))
                )

                torch.save(self.model.state_dict(), self.path_to_model)
                torch.save(self.optim.state_dict(), self.path_to_optimizer)

    def train(self, dataset, n_epoch=1, log_interval=10, batch_size_train=64):
        self.model.train()
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size_train, shuffle=True
        )

        for epoch in range(n_epoch):
            self.train_epoch(epoch, train_loader, log_interval)

    def validate(self, dataset, batch_size_test=1024):
        self.model.eval()

        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size_test, shuffle=True
        )

        test_loss = 0
        correct = 0

        test_losses = []

        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print(
            '\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )
        pd.DataFrame(
            {
                'metric_name': ['avg_loss', 'accuracy'],
                'value': [test_loss, 100.0 * correct / len(test_loader.dataset)],
            }
        ).to_csv(self.path_to_metrics)

    def save_model(self, path=None):
        torch.save(self.model.state_dict(), path or self.path_to_model)

    def load_model(self, path=None):
        self.model.load_state_dict(torch.load(path or self.path_to_model))
