# coding=utf-8
# @Time:2022/1/20下午9:28 
# @Author: 放羊Wa
# @Github: juliusyang97

# 1、函数forward()中，有一个变量w。这个变量最终的值是从for循环中传入的。
# 2、for循环中，使用了np.arange()。若对numpy不太熟悉，传送门Numpy数据计算从入门到实战(https://www.bilibili.com/video/BV1U7411x76j?p=5)
# 3、python中zip()函数的用法

import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


w_list = []
mse_list = []
for w in np.arange(0.0, 4.1, 0.1):
    print("w=", w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print('MSE=', l_sum / 3)
    w_list.append(w)
    mse_list.append(l_sum / 3)

plt.plot(w_list, mse_list)
plt.ylabel("Loss")
plt.xlabel("w")
plt.show()
