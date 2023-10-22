import fire
import hydra
from hydra.core.config_store import ConfigStore

from mlops_course.config import Params
from mlops_course.data import get_test_dataset, get_train_dataset
from mlops_course.model.model import Net
from mlops_course.trainer.mnist_solver import MnistSolver


cs = ConfigStore.instance()
cs.store(name="params", node=Params)


class Starter:
    def __init__(self):
        with hydra.initialize(config_path="../conf", version_base="1.3"):
            self.params: Params = hydra.compose(config_name="config")

    def train(self):
        net = Net()
        train_dataset = get_train_dataset()

        solver = MnistSolver(net)
        solver.train(
            train_dataset,
            n_epoch=self.params.model.n_epoch,
            log_interval=10,
            batch_size_train=self.params.model.train_batch_size,
        )

    def infer(self):
        net = Net()
        test_dataset = get_test_dataset()

        solver = MnistSolver(net)
        solver.load_model(self.params.paths.model)
        solver.validate(test_dataset, batch_size_test=self.params.model.test_batch_size)


#
# class MLOpsPractice(object):
#     def __init__(self):
#         with hydra.initialize(config_path="../conf", version_base="1.3"):
#             self.params: Params = hydra.compose(config_name="config")
#         torch.manual_seed(self.params.model.random_seed)
#
#     def train(self, plot: bool = False):
#         train_dataset, val_dataset = get_train_val_datasets(
#             dataset_path=self.params.path.fashion_mnist_train,
#             train_ratio=self.params.dataset.train_ratio,
#         )
#         model = MultiLabelClassifier()
#         trainer = ClassifierTrainer(
#             model=model,
#             batch_size=self.params.model.batch_size,
#             n_epoch=self.params.model.n_epoch,
#             lr=self.params.model.optim.lr,
#             weight_decay=self.params.model.optim.weight_decay,
#         )
#         trainer.fit(
#             train_dataset=train_dataset,
#             val_dataset=val_dataset,
#         )
#
#         if plot:
#             trainer.plot()
#         trainer.save_model(self.params.path.model)
#
#     def infer(self):
#         test_dataset = get_test_dataset(
#             dataset_path=self.params.path.fashion_mnist_test,
#         )
#         model = MultiLabelClassifier()
#         trainer = ClassifierTrainer(
#             model=model,
#             batch_size=self.params.model.batch_size,
#             n_epoch=self.params.model.n_epoch,
#             lr=self.params.model.optim.lr,
#             weight_decay=self.params.model.optim.weight_decay,
#         )
#         trainer.load_model(self.params.path.model)
#         trainer.process_val(test_dataset, output_csv=self.params.path.result)
#
#     def train_infer(self):
#         self.train()
#         self.infer()


def main():
    fire.Fire(Starter())


if __name__ == "__main__":
    main()
