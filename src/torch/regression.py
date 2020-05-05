from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x):
        x = self.layer(x)
        return x


def visualize_data():
    x = np.random.rand(100)
    y = np.sin(x) * np.power(x, 3) + 3 * x + np.random.rand(100) * 0.8

    plt.scatter(x, y)
    plt.show()


def train_model():
    x = np.random.rand(100)
    print(x)
    y = np.sin(x) * np.power(x, 3) + 3 * x + np.random.rand(100) * 0.8
    print(y)

    x = torch.from_numpy(x.reshape(-1, 1)).float()
    y = torch.from_numpy(y.reshape(-1, 1)).float()
    print(x)
    print(y)

    net = Net()
    print(net)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
    loss_func = torch.nn.MSELoss()

    inputs = Variable(x)
    outputs = Variable(y)

    for i in range(250):
        prediction = net(inputs)
        loss = loss_func(prediction, outputs)
        print('loss %s' % loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            # plot and show learning process
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=2)
            plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 10, 'color': 'red'})
            plt.pause(0.1)

    plt.show()
    print('------------DONE------------')


if __name__ == '__main__':
    train_model()
