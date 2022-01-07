# coding=utf-8
# @Time:2022/1/7下午10:43 
# @Author: 放羊Wa
# @Github: juliusyang97
import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn

train_dataset = torchvision.datasets.CIFAR10(root="../dataset", train=True, download=True,
                                             transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(train_dataset, batch_size=64)

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

print(vgg16_false)

vgg16_false.classifier.add_module("add_linear", nn.Linear(1000, 10))
print(vgg16_false)

vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)
