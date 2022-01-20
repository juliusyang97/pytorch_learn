# coding=utf-8
# @Time:2022/1/8上午9:55 
# @Author: 放羊Wa
# @Github: juliusyang97
import torch
import torchvision
from p26_model_save import *


# 方式1 -> 使用保存方式1来加载模型
model = torch.load("../models/vgg16_method1.pth")
# print(model)

# 方式2 -> 使用保存方式2来加载模型
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("../models/vgg16_method2.pth"))
# model = torch.load("../models/vgg_method2.pth")
# print(vgg16)


# 陷阱 --  使用方式1加载
# 这种方式需要加载时把 Tudui 这个类加载进来，但是不需要再创建（tudui = Tudui()）

# 解决方式1 -- 把class Tudui(模型的定义) 复制过来
# class Tudui(nn.Module):
#     def __init__(self):
#         super(Tudui, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x

# 解决方式2 -- 直接从源文件中导入class
# from p26_model_save import *


model = torch.load("../models/tudui_method1.pth")
print(model)

