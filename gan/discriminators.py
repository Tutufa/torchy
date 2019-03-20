import torch
import torch.nn as nn
from torch.nn import functional


class MNISTDiscriminator(nn.Module):
    def __init__(self, capacity: int=32):
        super(MNISTDiscriminator, self).__init__()

        self.capacity = capacity

        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=capacity, kernel_size=4, stride=2, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=capacity, out_channels=capacity*2, kernel_size=4, stride=2, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=capacity*2, out_channels=capacity*4, kernel_size=3, stride=2, padding=1)
        self.conv_4 = nn.Conv2d(in_channels=capacity*4, out_channels=1, kernel_size=4, stride=1, padding=0)

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        x = functional.leaky_relu(self.conv_1(in_tensor), 0.1)
        x = functional.leaky_relu(self.conv_2(x), 0.1)
        x = functional.leaky_relu(self.conv_3(x), 0.1)

        x = torch.sigmoid(self.conv_4(x))
        return x.view((-1, 1))
