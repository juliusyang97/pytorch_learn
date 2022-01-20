# coding=utf-8
# @Time:2022/1/20下午10:42 
# @Author: 放羊Wa
# @Github: juliusyang97

import matplotlib.pyplot as plt


# prepare the training set
x_data = [1., 2., 3.]
y_data = [2., 4., 6.]

# initial guess of weight
w = 1


# define the model linear model y = w*x
def forward(x):
    return x * w


# define the cost function MSE
def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)


# define the gradient function  gd
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


epoch_list = []
cost_list = []

print("Predit (before training)", 4, forward(4))
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val
    print('Epoch:', epoch, "w=", w, 'loss=', cost_val)

    epoch_list.append(epoch)
    cost_list.append(cost_val)

print('Predit (after training)', 4, forward(4))

plt.plot(epoch_list, cost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()