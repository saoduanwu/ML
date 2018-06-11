#!/usr/bin python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from numpy import *


class LinearRegression:
    def __init__(self, x, y, m, alpha, iterations):
        self.x = np.vstack((mat(ones((1, m))), x)).T
        self.y = y
        self.m = m
        self.theta = mat(zeros((2, 1)))
        self.alpha = alpha
        self.iterations = iterations

    def costFunction(self):
        h = self.x * self.theta
        J = (h - self.y.T).T * (h - self.y.T) / (2 * self.m)
        return J[0, 0]

    def gradientDescent(self):
        J = mat(zeros((1, self.iterations)))
        for i in range(self.iterations):
            J[0, i] = self.costFunction()
            delta = self.x.T * (self.x * self.theta - y.T) / self.m
            self.theta += -alpha * delta
        return self.theta, J

    def plotLearningCurve(self, J):
        plt.plot(mat(range(self.iterations)).T, J.T, '-r')
        plt.ylabel('costFunction J')
        plt.xlabel('nummber of iterate')
        plt.title('learning curve')
        plt.show()

    def plotGoodnessOfFit(self, theta):
        plt.plot(self.x[:, 1], y.T, 'og', ms=5, label='training data')
        plt.plot(self.x[:, 1], self.x * theta, '-r', label='fit curve')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('goodness of fit')
        plt.show()


if __name__ == '__main__':
    data = np.loadtxt('./ex1data1.txt', delimiter=',')
    # calculate with Matrix
    x = mat(data[:, 0])
    y = mat(data[:, 1])
    m = np.size(x, axis=1)
    alpha = 0.01
    iterations = 1500
    lr = LinearRegression(x, y, m, alpha, iterations)
    lr.costFunction()
    theta, J = lr.gradientDescent()
    lr.plotLearningCurve(J)
    lr.plotGoodnessOfFit(theta)
