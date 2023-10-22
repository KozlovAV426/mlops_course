from mlops_course.data import get_train_dataset
from mlops_course.trainer.mnist_solver import MnistSolver
from model.model import Net


def main():
    model = Net()
    train_dataset = get_train_dataset()

    solver = MnistSolver(model)
    solver.train(train_dataset)


if __name__ == "__main__":
    main()
