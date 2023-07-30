# -*- coding: gbk -*-

import numpy as np #����numpy�������ɾ���
import matplotlib.pyplot as plt #����matplotlib���е�pyplot���ڻ�ͼ

def sin_2pi(x):#�������sin��2 pi x������
    return np.sin(2*np.pi*x)

def get_traindata():#���ѵ������
    x = np.linspace(0, 1, num = 10)#����10�������㣬���ȷֲ�������[0,1]֮�䣬Ҫ�����ݽ���ʱʹ����һ�����ΪҪ�����ݽ϶�����ʱʹ��
    x = np.random.rand(200)
    y = sin_2pi(x) + np.random.normal(0, 0.1, 200)#���Ӹ�˹����������ѡȡ0.1
    return x,y

def matrix(x, m = 3):#�����Ա���x����X���󣬵�i��Ϊx��i-1�η���m��ʾ������Ĭ�ϳ�ʼֵΪ3
    X = pow(x, 0)#��һ��ȫΪ1
    for i in range(1 ,m + 1):
        X = np.column_stack((X, pow(x, i)))#��ÿһ��ƴ����������չ��ɾ���
    return X

def get_params(X, Y):#��С���˷������ͷ��ͨ����ʽ�õ�����w��XΪƴ�ӵõ��ľ���YΪ���������yֵ�ľ���
    w = np.linalg.inv((X.T).dot(X)).dot(X.T).dot(Y) #��ʽW* = (X'��X)^(-1) (X')��Y=X^(-1)��Y
    return w

def get_params_with_penalty(X, Y, l):#��С���˷����ͷ��ͨ����ʽ�õ�����w��XΪƴ�ӵõ��ľ���YΪ���������yֵ�ľ���lΪ��Ϊ�趨�ĳ�����
    w = np.linalg.inv((X.T).dot(X) + l * (np.identity(len(X.T)))).dot(X.T).dot(Y) #��ʽ:w* = (X'��X+lI)^(-1) (X')��Y
    return w

def gradient_descent_fit(X, Y, lr = 0.01):#�ݶ��½���������Ž⣬ͨ����ʽ�õ�����w��XΪƴ�ӵõ��ľ���YΪ���������yֵ�ľ���lrΪѧϰ��
    w = ([ 0, 0, 0, 0, 0, 0])#��ʼ��w
    for i in range(100000):#�����㹻��Ĳ������м��㣬Ҳ�������ݶ�С��ĳ����Сֵ���м���
        Y1 = np.dot(X, w)
        grad = np.dot(X.T, Y1-Y)
        w -= grad * lr
    return w

def conjugate_gradient_fit(X, Y, m):#�����ݶȷ�������Ž⣬ͨ����ʽ�õ�����w��XΪƴ�ӵõ��ľ���YΪ���������yֵ�ľ���mΪ����
    A = np.dot(X.T, X)#A=X^T��X������A�Ѿ�����
    b = np.dot(X.T, Y)#b=X^T��Y
    w = np.zeros(m + 1)#��ʼ��w
    epslon = 0.00001#��һ����С����Ϊ��������
    d = r = b - np.dot(A, w)#����������ʽ���г�ʼ��
    r0=r
    while True:#��������
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
plt.plot(x, y, color = 'green')#����ԭ������ͼ����������ɫ����
x1, y1 = get_traindata()
plt.scatter(x1, y1, color = 'red')#�����������ݵ�ɢ��ͼ��ɢ���ú�ɫ����
X = matrix(x1, m)
Y = y1
w = get_params(X, Y)#������ֵ�õ�����w
X1 = matrix(x, m)#����������Ҫ��������ݣ�ʹ��ԭ�ȶ���100��x��ֵ�õ��������
Y1 = X1.dot(w)
plt.plot(x, Y1, label = "m = " + str(m))#�����������ͼ��
plt.legend()
plt.show()