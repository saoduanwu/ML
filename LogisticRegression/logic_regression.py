#!/usr/bin python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def loadData():
    data = np.loadtxt('./ex2data1.txt', delimiter=',')
    x = data[:, :-1]
    y = data[:, -1]
    return x, y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def costFunction(x, y, theta):
    h = sigmoid(x.dot(theta))
    J = -(y.dot(np.log(h)) + (1 - y).dot(np.log(1 - h))) / m
    return J


def gradientDescent(x, y, alpha, theta, iterations):
    J = np.zeros((iterations,))
    for i in range(iterations):
        J[i] = costFunction(x, y, theta)
        delta = x.T.dot(sigmoid(x.dot(theta)) - y) / m
        theta += -alpha * delta
    return theta, J


def learningCurve(iterations, J):
    plt.plot(np.arange(iterations), J, '-r')
    plt.xlabel('nummber of iterate')
    plt.ylabel('cost function J')
    plt.title('learning curve')
    plt.show()


def decisionBoundary(x, y, theta):
    positive = np.where(y == 1)
    negative = np.where(y == 0)
    print(x[positive, 1])
    # plt.plot(x[positive,1].T,x[positive,2].T,'r+',label='positive')
    p = plt.scatter(x[positive, 1], x[positive, 2], c='g', marker='o', s=30)
    # plt.plot(x[negative,1].T,x[negative,2].T,'go',label='negative')
    n = plt.scatter(x[negative, 1], x[negative, 2], c='r', marker='+', s=30)
    print(x[positive, 1])
    plot_x = np.arange(np.min(x[:, 1]), np.max(x[:, 1]))
    plot_y = -(theta[0] + theta[1] * plot_x) / theta[2]
    plt.plot(plot_x, plot_y, 'm-')
    plt.legend([p, n], ['positive', 'negative'], loc='upper right')
    plt.xlabel('x1')
    plt.xlabel('x2')
    plt.show()


# def plotDecisionBoundary(theta, x, y):
#     pos = np.where(y == 1)
#     neg = np.where(y == 0)
#     p1 = plt.scatter(x[pos, 1], x[pos, 2], marker='+', s=60, color='r')
#     p2 = plt.scatter(x[neg, 1], x[neg, 2], marker='o', s=60, color='y')
#     plot_x = np.array([np.min(x[:, 1])-2, np.max(x[:, 1]+2)])
#     print(x[pos, 1])
#     print(plot_x)
#     plot_y = -1/theta[2]*(theta[1]*plot_x+theta[0])
#     print(plot_y)
#     plt.plot(plot_x, plot_y)
#     plt.legend((p1, p2), ('Admitted', 'Not admitted'), loc='upper right', fontsize=8)
#     plt.xlabel('Exam 1 score')
#     plt.ylabel('Exam 2 score')
#     plt.show()

def predict():
    p = np.zeros((m,))
    posstive = np.where(x.dot(theta) > 0)
    negative = np.where(x.dot(theta) < 0)
    p[posstive] = 1
    p[negative] = 0
    print('Train Accuracy: ', np.sum(p == y) / m)
    # return p


if __name__ == '__main__':
    loadData()
    x, y = loadData()
    m, n = x.shape
    x = np.hstack((np.ones((m, 1)), x))
    theta = np.zeros((n + 1,))
    costFunction(x, y, theta)
    alpha = 0.001
    iterations = 200000
    theta, J = gradientDescent(x, y, alpha, theta, iterations)

    learningCurve(iterations, J)
    # theta=np.array([-7.45017822,0.06550395,0.05898701])
    decisionBoundary(x, y, theta)
    # plotDecisionBoundary(theta, x, y)
    p = predict()
    # print(p)
