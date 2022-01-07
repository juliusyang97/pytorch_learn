# coding=utf-8
# @Time:2022/1/7下午9:12 
# @Author: 放羊Wa
# @Github: juliusyang97
import torch
from torch.nn import L1Loss
from torch import nn

input = torch.tensor([1, 2, 3], dtype=torch.float32)
target = torch.tensor([1, 2, 5], dtype=torch.float32)

input = torch.reshape(input, (1, 1, 1, 3))
target = torch.reshape(target, (1, 1, 1, 3))

# L1Loss
loss = L1Loss(reduction="sum")
result = loss(input, target)

# MSELoss
loss_mse = nn.MSELoss()
result_mse = loss_mse(input, target)

print(result)
print(result_mse)

# CrossEntropyLoss
x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)



