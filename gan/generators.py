import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate


class MNISTGenerator(nn.Module):
    def __init__(self):
        super(MNISTGenerator, self).__init__()

        self.dense_1 = nn.Linear(in_features=96, out_features=1024)
        self.bn_1 = nn.BatchNorm1d(1024)

        self.dense_2 = nn.Linear(in_features=1024, out_features=6272)
        self.bn_2 = nn.BatchNorm1d(6272)

        self.conv_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_1_bn = nn.BatchNorm2d(64)

        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, in_tensor: torch.Tensor):
        x = F.relu(self.bn_1(self.dense_1(in_tensor)))
        x = F.relu(self.bn_2(self.dense_2(x)))

        x = x.view((-1, 128, 7, 7))
        
        x = interpolate(input=x, scale_factor=2)
        x = F.relu(self.conv_1_bn(self.conv_1(x)))

        x = interpolate(input=x, scale_factor=2)
        x = self.conv_2(x)
        
        return torch.sigmoid(x)























