#!/usr/bin python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

def dataLoad():
    data = np.loadtxt('ex2data2.txt',delimiter=',')
    x = data[:,:-1]
    y = data[:,-1]
    # m,n = x.shape
    # one = np.ones((m,1))
    # x = np.hstack((one,x))
    # print(one)
    return x,y

def dataSetDis(x,y):
    pos = np.where(y==1)
    neg = np.where(y==0)
    posditive = plt.scatter(x[pos,0],x[pos,1],s=30,c='r',marker='o')
    negative = plt.scatter(x[neg,0],x[neg,1],s=30,c='g',marker='+')
    plt.legend([posditive,negative],[posditive,negative],loc='upper left ')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def mapFeature(x1, x2):
    degree = 6
    col = int(degree*(degree+1)/2+degree+1)
    out = np.ones((np.size(x1, 0), col))
    count = 1
    for i in range(1, degree+1):
        for j in range(i+1):
            out[:, count] = np.power(x1, i-j)*np.power(x2, j)
            count += 1
    return out


def sigmoid(z):
    h = 1/(1+np.exp(-1*z))
    return h





def gradientDecsent(x,y,alpha,theta,iterations,m):
    h = sigmoid(x.dot(theta))
    J=np.zeros((iterations,))
    print(J)
    grad = np.zeros((np.size(x,axis=1),))
    for i in range(iterations):
        J[i] = costFunction(x,y,theta,lamd)
        grad[0] = x[:,0].dot(h-y)/m
        grad[1:] = x[:,1:].T.dot(h-y)/m +lamd*theta[1:]/m
        theta += -alpha*grad
    return theta,J

def costFunction(theta,x,y,lamd):
    z=x.dot(theta)
    h = sigmoid(z)
    J = -(y.dot(np.log(h))+(1-y).dot(np.log(1-h)))/m+lamd*theta[1:].dot(theta[1:])/(2*m)
    return J

def gradient(theta,x,y,lamd):
    # h = sigmoid(x.dot(theta))
    # print(x.shape)
    # print(theta.shape)
    z=x.dot(theta)
    h = sigmoid(z)
    grad = np.zeros((np.size(theta,axis=0),))
    grad[0] = x[:,0].dot(h-y)/m
    grad[1:] = x[:,1:].T.dot(h-y)+lamd*theta[1:]/m
    return grad


def predict(x):
    m = np.size(x,axis=0)
    p = np.zeros((m,))
    z = x.dot(theta)
    # print(z)
    pos = np.where(z>=0)
    meg = np.where(z<0)
    p[pos] = 1
    p[meg] = 0
    print('accuracy=%s'%(np.sum(p==y)/m))

# def predict(theta, x):
#     m = np.size(x, 0)
#     p = np.zeros((m,))
#     pos = np.where(x.dot(theta) >= 0)
#     neg = np.where(x.dot(theta) < 0)
#     p[pos] = 1
#     p[neg] = 0
#     return p



if __name__ == '__main__':
    x,y = dataLoad()
    # print(x)
    # print(y)
    # dataSetDis(x,y)
    x = mapFeature(x[:, 0], x[:, 1])
    # print(x)
    m,n = x.shape
    theta = np.zeros((n,))
    theta = np.zeros((np.size(x, 1),))
    lamd = 1
    J = costFunction(theta,x, y,lamd)
    print(J)
    theta = np.zeros((np.size(x, 1),))
    print(theta)
    print(x.shape)
    print('8'*80)
    res = op.minimize(costFunction,x0=theta,method="BFGS",jac=gradient,args=(x,y,lamd))
    theta = res.x
    print('theta=%s'%theta)


    # alpha=0.001
    # iterations = 10000
    # theta,J = gradientDecsent(x, y, alpha, theta, iterations, m)
    # print(J)
    # print(theta)
    predict(x)

    # p = predict(theta, x)
    # print('Train Accuracy: ', np.sum(p == y) / np.size(y, 0))