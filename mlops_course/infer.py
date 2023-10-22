from mlops_course.data import get_test_dataset
from mlops_course.trainer.mnist_solver import MnistSolver
from model.model import Net


def main():
    model = Net()
    test_dataset = get_test_dataset()

    solver = MnistSolver(model)
    solver.load_model()
    solver.validate(test_dataset)


if __name__ == "__main__":
    main()
