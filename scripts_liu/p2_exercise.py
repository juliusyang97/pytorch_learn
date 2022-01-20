# coding=utf-8
# @Time:2022/1/20下午10:11 
# @Author: 放羊Wa
# @Github: juliusyang97

# Numpy
import numpy
# For plotting
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


def forward(w: numpy.ndarray, b: numpy.ndarray, x: float) -> numpy.ndarray:
    return w * x + b


def loss(y_hat: numpy.ndarray, y: float) -> numpy.ndarray:
    return (y_hat - y) ** 2


w_cor = numpy.arange(0.0, 4.0, 0.1)
b_cor = numpy.arange(-2.0, 2.1, 0.1)

# 此处直接使用矩阵进行计算
w, b = numpy.meshgrid(w_cor, b_cor)
mse = numpy.zeros(w.shape)

for x, y in zip(x_data, y_data):
    _y = forward(w, b, x)
    mse += loss(_y, y)
mse /= len(x_data)

h = plt.contourf(w, b, mse)

fig = plt.figure()
ax = Axes3D(fig)
plt.xlabel(r'w', fontsize=20, color='cyan')
plt.ylabel(r'b', fontsize=20, color='cyan')
ax.plot_surface(w, b, mse, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
plt.show()





