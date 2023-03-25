import torch
from torch import nn
from torch.nn import functional as F

from torch.nn.modules.batchnorm import BatchNorm1d, BatchNorm2d
import numpy as np


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5,5))

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=9, kernel_size=(3,3))
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))
        self.bn_conv2 = BatchNorm2d(9)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(13 * 13 * 9, 512)
        self.bn1 = BatchNorm1d(512)

        self.fc_hdn = nn.Linear(512, 128)
        self.bn2 = BatchNorm1d(128)

        self.fc_out = nn.Linear(128, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.bn_conv2(x)

        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.bn1(x)

        x = F.relu(self.fc_hdn(x))
        x = self.bn2(x)

        x = self.fc_out(x)

        return x
