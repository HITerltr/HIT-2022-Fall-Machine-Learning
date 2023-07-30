# -*- coding: gbk -*-

import numpy as np #调用numpy用于生成矩阵
import matplotlib.pyplot as plt #调用matplotlib库中的pyplot用于绘图

def sin_2pi(x):#打包利用sin（2 pi x）函数
    return np.sin(2*np.pi*x)

def get_traindata():#获得训练数据
    x = np.linspace(0, 1, num = 10)#共计10个样本点，均匀分布在区间[0,1]之间，要求数据较少时使用这一项，下面为要求数据较多的情况时使用
    x = np.random.rand(200)
    y = sin_2pi(x) + np.random.normal(0, 0.1, 200)#增加高斯噪声，方差选取0.1
    return x,y

def matrix(x, m = 3):#根据自变量x生成X矩阵，第i列为x的i-1次方，m表示阶数，默认初始值为3
    X = pow(x, 0)#第一列全为1
    for i in range(1 ,m + 1):
        X = np.column_stack((X, pow(x, i)))#将每一列拼接起来就扩展变成矩阵
    return X

def get_params(X, Y):#最小二乘法不带惩罚项，通过公式得到参数w，X为拼接得到的矩阵，Y为样本因变量y值的矩阵
    w = np.linalg.inv((X.T).dot(X)).dot(X.T).dot(Y) #公式W* = (X'·X)^(-1) (X')·Y=X^(-1)·Y
    return w

def get_params_with_penalty(X, Y, l):#最小二乘法带惩罚项，通过公式得到参数w，X为拼接得到的矩阵，Y为样本因变量y值的矩阵，l为人为设定的超参数
    w = np.linalg.inv((X.T).dot(X) + l * (np.identity(len(X.T)))).dot(X.T).dot(Y) #公式:w* = (X'·X+lI)^(-1) (X')·Y
    return w

def gradient_descent_fit(X, Y, lr = 0.01):#梯度下降法求解最优解，通过公式得到参数w，X为拼接得到的矩阵，Y为样本因变量y值的矩阵，lr为学习率
    w = ([ 0, 0, 0, 0, 0, 0])#初始化w
    for i in range(100000):#采用足够多的步数进行计算，也可以用梯度小于某个极小值进行计算
        Y1 = np.dot(X, w)
        grad = np.dot(X.T, Y1-Y)
        w -= grad * lr
    return w

def conjugate_gradient_fit(X, Y, m):#共轭梯度法求解最优解，通过公式得到参数w，X为拼接得到的矩阵，Y为样本因变量y值的矩阵，m为阶数
    A = np.dot(X.T, X)#A=X^T·X，当成A已经正则
    b = np.dot(X.T, Y)#b=X^T·Y
    w = np.zeros(m + 1)#初始化w
    epslon = 0.00001#设一个极小数作为跳出条件
    d = r = b - np.dot(A, w)#按照上述公式进行初始化
    r0=r
    while True:#迭代部分
        al = np.dot(r.T, r) / np.dot(np.dot(d, A), d)
        w += al * d
        r1 = r - al*np.dot(A, d)
        be = np.dot(r1.T, r1) / np.dot(r.T, r)
        d = be * d + r1
        r = r1
        if np.linalg.norm(r) / np.linalg.norm(r0) < epslon:
            break
    return w

x = np.linspace(0, 1, num = 100)
y=sin_2pi(x)
m = 5
plt.plot(x, y, color = 'green')#绘制原函数的图像，曲线用绿色绘制
x1, y1 = get_traindata()
plt.scatter(x1, y1, color = 'red')#绘制样本数据的散点图，散点用红色绘制
X = matrix(x1, m)
Y = y1
w = get_params(X, Y)#用样本值得到参数w
X1 = matrix(x, m)#绘制曲线需要更多的数据，使用原先定义100个x的值得到拟合曲线
Y1 = X1.dot(w)
plt.plot(x, Y1, label = "m = " + str(m))#绘制拟合曲线图像
plt.legend()
plt.show()