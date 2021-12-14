import torch
from torch import nn
from typing import Tuple


class Neroro(nn.Module):
    def __init__(self):
        super(Neroro, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 10)
        self.activation = nn.LeakyReLU(0.04)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer1(x.view(-1, 28 * 28))
        out = self.activation(out)
        return out


def calc_convolution_output(input_size: int, kernel_size: int, padding: int = 0, stride: int = 1,
                            dilation: int = 1):
    return int(((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)

