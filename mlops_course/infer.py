import pandas as pd
import torch
import torch.nn.functional as F

from config import PATH_TO_METRICS, PATH_TO_MODEL
from data import test_loader, train_loader
from model.model import Net


network = Net()
network.load_state_dict(torch.load(PATH_TO_MODEL))
n_epochs = 3

test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
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
    ).to_csv(PATH_TO_METRICS)


if __name__ == "__main__":
    test()
