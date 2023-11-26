import pytorch_lightning as pl
import torch
import torchvision


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        train_ratio: float,
        batch_size: int,
        num_workers: int = 3,
    ):
        super().__init__()

        self.dataset_path = dataset_path or '../data/.files'
        self.num_workers = num_workers

        self.train_ratio = train_ratio
        self.batch_size = batch_size

        self.train_dataset, self.val_dataset = self.split_into_train_val_dataset()
        self.test_dataset = self.get_test_dataset()

    def split_into_train_val_dataset(self):
        print("Loading train-val dataset")

        dataset = torchvision.datasets.MNIST(
            self.dataset_path,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )

        print(f"dataset type = {type(dataset)}")

        return dataset, dataset

    def get_test_dataset(self):
        return torchvision.datasets.MNIST(
            self.dataset_path,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )

    def make_data_loader(self, dataset, shuffle=False):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def train_dataloader(self):
        return self.make_data_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.make_data_loader(self.val_dataset)

    def test_dataloader(self):
        return self.make_data_loader(self.test_dataset)


def get_train_dataset():
    return torchvision.datasets.MNIST(
        '../data/.files',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )


def get_test_dataset():
    return torchvision.datasets.MNIST(
        '../data/.files',
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )
