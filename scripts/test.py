# coding=utf-8
# @Time:2022/1/8下午5:22 
# @Author: 放羊Wa
# @Github: juliusyang97

import torch
import torchvision
from PIL import Image
from torch import nn
from model import *

img_path = "../imgs/dog.png"
img = Image.open(img_path)
print(img)
img = img.convert("RGB")  # png格式图片有四个通道，除了RGB还有一个透明度；调用这步可以保留其颜色通道。
                          # 如果图片本来就是三通道，加上此操作，不影响。以后可以直接加上，以适应所有图片。
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

img = transform(img)
print(img.shape)

# 导入模型 -- 由于模型保存使用的方式1，所以需要导入模型
# from model import *

model = torch.load("../models/tudui_4.pth", map_location=torch.device("cpu"))
print(model)

img = torch.reshape(img, (1, 3, 32, 32))

model.eval()
with torch.no_grad():
    output = model(img)
print(output)

print(output.argmax(1))

