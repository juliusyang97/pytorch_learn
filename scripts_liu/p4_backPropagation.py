# coding=utf-8
# @Time:2022/1/21上午9:33 
# @Author: 放羊Wa
# @Github: juliusyang97

# 【PyTorch】Tensor和tensor的区别:https://blog.csdn.net/tfcy694/article/details/85338745

# 1. w是Tensor(张量类型)，Tensor中包含data和grad，data和grad也是Tensor。\
#    grad初始为None，调用l.backward()方法后w.grad为Tensor，故更新w.data时需使用w.grad.data。\
#    如果w需要计算梯度，那构建的计算图中，跟w相关的tensor都默认需要计算梯度。

# a = torch.tensor([1.0])
# a.requires_grad = True  # 或者 a.requires_grad_()
# print(a)
# print(a.data)
# print(a.type())         # a的类型是tensor
# print(a.data.type())    # a.data 的类型是tensor
# print(a.grad)
# print(type(a.grad))


# 2. w是Tensor， forward函数的返回值也是Tensor，loss函数的返回值也是Tensor;
# 3. 本算法中反向传播主要体现在，l.backward()。调用该方法后w.grad由None更新为Tensor类型，\
#    且w.grad.data的值用于后续w.data的更新。
#    l.backward()会把计算图中所有需要梯度(grad)的地方都会求出来，然后把梯度都存在对应的待求的参数中，最终计算图被释放。
#    取tensor中的data是不会构建计算图的。


import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.tensor([1.0])  # w的初始值为1.0
w.requires_grad = True  # 计算梯度,默认不计算


def forward(x):
    return x * w  # w 是一个tensor


def loss(x, y):  # 构建计算图
    y_pred = forward(x)
    return (y_pred - y) ** 2


print('Predit (before training)', 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)  # l是一个张量，tensor主要是在建立计算图 forward, compute the loss
        l.backward()    # backward,compute grad for Tensor whose requires_grad set to True
        print('\t grad:', x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data   # 权重更新时，注意这里的grad是一个tensor，所以要取他的data

        w.grad.data.zero_()  # 释放之前计算的梯度

    print('progress:', epoch, l.item())  # 取出loss使用l.item，不要直接使用l（l是tensor会构建计算图）

print('Predit (after training)', 4, forward(4).item())




