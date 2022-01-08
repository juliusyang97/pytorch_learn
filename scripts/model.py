# coding=utf-8
# @Time:2022/1/8上午11:26 
# @Author: 放羊Wa
# @Github: juliusyang97
import torch
from torch import nn


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

if __name__ == "__main__":
    tudui = Tudui()
    input = torch.ones((64, 3, 32, 32))
    output = tudui(input)
    print(output.shape)
