# coding=utf-8
# @Time:2022/1/7下午10:13 
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
optimer = torch.optim.SGD(tudui.parameters(), lr=0.01, momentum=0.9) # 定义优化器
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        result_loss = loss(outputs, targets)
        optimer.zero_grad() # 优化器中每一个梯度参数清零
        result_loss.backward() # 反向传播求梯度
        optimer.step() # 对参数进行调优
        running_loss = running_loss + result_loss
    print(running_loss)