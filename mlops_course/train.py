import os

import torch
import torch.nn.functional as F
import torch.optim as optim

from config import PATH_TO_MODEL, PATH_TO_OPTIMIZER
from data import train_loader
from model.model import Net


n_epochs = 1
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1

torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_losses = []
train_counter = []

network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

os.makedirs(PATH_TO_MODEL.split('/')[0], exist_ok=True)
os.makedirs(PATH_TO_OPTIMIZER.split('/')[0], exist_ok=True)


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

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
            torch.save(network.state_dict(), PATH_TO_MODEL)
            torch.save(optimizer.state_dict(), PATH_TO_OPTIMIZER)


if __name__ == "__main__":
    train(1)
