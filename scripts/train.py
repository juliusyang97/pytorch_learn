# coding=utf-8
# @Time:2022/1/8上午10:35 
# @Author: 放羊Wa
# @Github: juliusyang97

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *


# 准备数据集
train_dataset = torchvision.datasets.CIFAR10(root="../dataset", train=True, download=True,
                                             transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10(root="../dataset", train=False, download=True,
                                            transform=torchvision.transforms.ToTensor())

# 使用DataLoader来加载数据集
train_dataloader = DataLoader(train_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)

# 数据集的长度 -- length
train_data_size = len(train_dataset)
test_data_size = len(test_dataset)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 创建网络模型
# 方式1：从模型文件导入（推荐）
# from model import *

# 方式2：直接定义网络模型
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
        x= self.model1(x)
        return x

tudui= Tudui()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练轮数
epoch = 5

# add tensorboard logs
writer = SummaryWriter(log_dir="../logs/train")

for i in range(epoch):
    print("--------第{}轮训练开始------".format(i+1))

    # 训练步骤开始
    tudui.train()  # 网络模型模式设置，pytorch官网有讲解，对Dropout、BN等起作用。
    for data in train_dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), global_step=total_train_step)

    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的准确率：{}".format(total_accuracy / test_data_size))
    writer.add_scalar(tag="test_loss", scalar_value=total_test_loss, global_step=total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_loss += 1

    torch.save(tudui, "../models/tudui_{}.pth".format(i))
    # torch.save(tudui.state_dict(), "../models/tudui_{}.pth".format(i))
    print("模型已保存")
writer.close()

print("训练测试完成")

# 出现问题：
# 1.RuntimeError: Integer division of tensors using div or / is no longer supported, and in a future release div will perform true division as in Python 3. Use true_divide or floor_divide (// in Python) instead.
# 2.测试准确率一直为零