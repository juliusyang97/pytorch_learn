# coding=utf-8
# @Time:2022/1/7下午11:13 
# @Author: 放羊Wa
# @Github: juliusyang97
import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式1 -- 模型结构+模型参数
# torch.save(vgg16, "../models/vgg16_method1.pth")

# 保存方式2 -- 模型参数（官方推荐）
torch.save(vgg16.state_dict(), "../models/vgg16_method2.pth")

# 陷阱 -- 使用方式1保存的陷阱
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x

tudui = Tudui()
torch.save(tudui, "../models/tudui_method1.pth")

