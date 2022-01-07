# coding=utf-8
# @Time:2022/1/7下午9:40 
# @Author: 放羊Wa
# @Github: juliusyang97
import torch
import torchvision
from torch.utils.data import DataLoader
from p22_nn_seq import Tudui
from torch import nn


dataset = torchvision.datasets.CIFAR10(root="../dataset", train=True,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=1)

tudui = Tudui()

loss = nn.CrossEntropyLoss()

for data in dataloader:
    imgs, targets = data
    outputs = tudui(imgs)
    result_loss = loss(outputs, targets)
    result_loss.backward()
    print(result_loss)

