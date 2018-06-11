#!/usr/bin python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def loadData():
    data = np.loadtxt('./ex1data2.txt', delimiter=',')
    x = data[:, :-1]
    y = data[:, -1]
    return x, y


# 特征值归一化处理
def featureNormalization(x):
    av = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)  # standard deviation
    x = np.divide(x - av, sigma)
    return x


# 损失函数
def costFunction(x, y, theta):
    h = x.dot(theta)
    J = (h - y).dot(h - y) / (2 * m)
    return J


# 梯度下降
def gradientDescent(x, y, theta, alpha, iterations):
    J = np.zeros((iterations,))
    for i in range(iterations):
        J[i] = costFunction(x, y, theta)
        delta = x.T.dot(x.dot(theta) - y) / m
        theta += -alpha * delta
    return theta, J


# 学习曲线
# def learningCurve(iterations, J):
def learningCurve(iterations, J):
    plt.plot(np.arange(iterations), J, '-b')
    plt.xlabel('nummber of iterate')
    plt.xlabel('cost function J')
    plt.title('learning curve')
    plt.show()


if __name__ == '__main__':
    # calculate with array
    x, y = loadData()
    m, n = x.shape
    x = featureNormalization(x)
    x = np.hstack((np.ones((m, 1)), x))
    theta = np.zeros((n + 1,))
    alpha = 0.01
    iterations = 400
    costFunction(x, y, theta)
    theta, J = gradientDescent(x, y, theta, alpha, iterations)
    print(J)
    print(theta)
    learningCurve(iterations, J)
