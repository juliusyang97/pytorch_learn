# coding=utf-8
# @Time:2022/1/7下午6:24 
# @Author: 放羊Wa
# @Github: juliusyang97
import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Linear

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

# for data in dataloader:
#     imgs, targets = data
#     print(imgs.shape)
#     output = torch.reshape(imgs, (1, 1, 1, -1))
#     print(output.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

tudui = Tudui()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    output = torch.flatten(imgs)
    print(output.shape)
    output = tudui(output)
    print(output.shape)
