# -*- coding: gbk -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#����Ϊ����õ�ȫ�ֱ���
sigma = 0.1#�������ݵı�׼��
n = 100#���������������
naiveFlag = False#if true,���ɵ�������������Bayes��������������Bayes
epsilon = 1e-5#������߽��ܷ�Χ�ڵ����ֵ
eta = 1e-1#ѧϰ��
lamda = 0#��������
epoch = 5000#ѵ������

def generate_Data():#�ڷ�Χ����[0,1]�����ɶ�ά���������� 
    a = n // 2#���A�����������
    b = n - a#���B�����������
    cov_xy = 0.08#����ά�ȵ�Э���������������Bayes�ļ��裩
    x_mean_1 = [-0.75, -0.25]#���1�ľ�ֵ
    x_mean_2 = [0.75, 0.25]#���2�ľ�ֵ
    train_x = np.zeros((n, 2))#train_x�б���������(data)
    train_y = np.zeros(n)#train_y�б���������ǩ(label)

    if naiveFlag:#����������Bayes�ļ���
        train_x[:a, :] = np.random.multivariate_normal(x_mean_1, [[sigma, 0], [0, sigma]], size = a)
        train_x[a:, :] = np.random.multivariate_normal(x_mean_2, [[sigma, 0], [0, sigma]], size = b)
        train_y[:a] = 0
        train_y[a:] = 1
    else:#������������Bayes�ļ���
        train_x[:a, :] = np.random.multivariate_normal(x_mean_1, [[sigma, cov_xy], [cov_xy, sigma]], size = a)
        train_x[a:, :] = np.random.multivariate_normal(x_mean_2, [[sigma, cov_xy], [cov_xy, sigma]], size = b)
        train_y[:a] = 0
        train_y[a:] = 1

    return train_x, train_y

def generate_X(train_x):#����X����
    raw_X = np.ones((train_x.shape[0], 1))
    for i in range(0, train_x.shape[1]):
        j = train_x[:, i]
        j = j.reshape((train_x.shape[0], 1))
        raw_X = np.hstack((raw_X, j))

    X = raw_X.T
    assert X.shape[0] == train_x.shape[1] + 1
    assert X.shape[1] == train_x.shape[0]

    return X  # ����������(3,500) (4, 49011)

def sigmoid(x):#����sigmoid����������Ϊ������������Ż�����
    if x >= 0:  # ��sigmoid�������������Ż���������ֹ����޷��ֲ����������
        return 1.0 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))

def gradient(X, train_y, w):#�����ݶ�
    grad = np.zeros((1, X.shape[0]))

    for i in range(0, X.shape[1]):
        wXT = w @ X[:, i]
        grad += X[:, i].T * (train_y[i] - sigmoid(wXT))

    return grad

def loss(X, train_y, w):#��ʧ�����������Ż�ʽ�ӣ���һ��
    loss = 0
    size = float(train_y.shape[0])

    for i in range(0, X.shape[1]):
        wXT = w @ X[:, i]
        loss += (train_y[i] * wXT / size - np.log(1 + np.exp(wXT)) / size)

    return loss

def gradient_descent(train_x, train_y):#�ݶ��½��㷨����������
    eta_temp = eta
    w = np.ones((1, train_x.shape[1] + 1))#(1,3)����һ��ȫ����Ϊ1
    X = generate_X(train_x)
    grad = gradient(X, train_y, w)
    loss_list = []
    loss_0 = 0
    loss_1 = loss(X, train_y, w)
    count = 0

    while count < epoch:#�趨���ֵ����С�ڸ����ֵʱ��ֹͣ����
        count += 1
        w = w - eta_temp * lamda * w + eta_temp * grad
        loss_0 = loss_1
        loss_1 = loss(X, train_y, w)

        if(np.abs(loss_1) > np.abs(loss_0)):#loss������������ѧϰ�ʼ��룬��ֹ���������в�������
            eta_temp *= 0.5
        grad = gradient(X, train_y, w)
        loss_list.append(np.abs(loss_0))
        print(count)

    return w, count, loss_list

def loss_show(loss_list):#չʾloss�����ı仯���
    epoch_list = [x for x in range(1, epoch, 1)]
    loss_list.remove(loss_list[0])
    plt.plot(epoch_list, loss_list, c='#DB4D6D', linewidth=2, label='loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    return 0

def accuracy(X, train_y, w):#�������׼ȷ��
    wrong = 0
    total = train_y.shape[0]

    for i in range(0, train_y.shape[0]):
        flag = train_y[i]
        value = w @ X[:, i]

        if((flag == 1 and value < 0) or (flag != 1 and value > 0)):
            wrong += 1
    temp = float(wrong) / total
    acc = temp if temp > 0.5 else 1 - temp

    return acc

def class_show(train_x, train_y, w, count):#���Ʒ���Ч��ͼ
    xa = []
    xb = []
    ya = []
    yb = []

    for i in range(0, train_x.shape[0]):
        if train_y[i] == 0:
            xa.append(train_x[i][0])
            ya.append(train_x[i][1])
        elif train_y[i] == 1:
            xb.append(train_x[i][0])
            yb.append(train_x[i][1])

    plt.scatter(xa, ya, c='red', s=10, label='class A')
    plt.scatter(xb, yb, c='blue', s=10, label='class B')#����ɢ��ͼ�����ղ�ͬ��ǩ������ɫ

    x_real = np.arange(-2, 2, 0.01)
    w0 = w[0][0]
    w1 = w[0][1]
    w2 = w[0][2]
    y_real = -(w1 / w2) * x_real - w0 / w2

    plt.plot(x_real, y_real, c='#DB4D6D', linewidth=2, label='fitting')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('data_size = ' + str(n) + ',' + 'lamda = ' + str(lamda) + ',' + 'naiveFlag = ' + str(naiveFlag))#�ݶ��½�
    plt.show()

    return 0

def gradient_descent_show():#���ݶ��½����������չʾ
    train_x, train_y = generate_Data()
    #print(train_x, train_y)
    #print(train_x.shape, train_y.shape)
    X = generate_X(train_x)
    w, count, loss_list = gradient_descent(train_x, train_y)
    acc = accuracy(X, train_y, w)
    print(w)
    print("acc:" + str(acc))
    class_show(train_x, train_y, w, count)
    loss_show(loss_list)

    return 0

def skin_read():#��ȡUCI Skin Segmentation���ݼ�
    skin_data = np.loadtxt('./Skin_NonSkin.txt', dtype=np.int32)
    np.random.shuffle(skin_data)#���Ҷ�������ݣ��Ա�ֳ�ѵ�����Ͳ��Լ�
    test_rate = 0.25#���Լ�����
    train_rate = 0.75#ѵ��������
    step = 20#��ȡ���������Ĳ���
    #print(skin_data.shape)#(245057, 4)
    #print(skin_data)
    new_data = skin_data[::step]
    n = new_data.shape[0]#��������
    raw_train_data = new_data[:int(test_rate * n), :]#ѵ����
    raw_test_data = new_data[int(test_rate * n):, :]#���Լ�
    train_x = raw_train_data[:, :-1] - 100
    train_y = raw_train_data[:, -1:] - 1
    test_x = raw_test_data[:, :-1] - 100
    test_y = raw_test_data[:, -1:] - 1
    #print(train_x.shape)#(980, 3)
    #print(train_y.shape)#(980, 1)

    return train_x, train_y, test_x, test_y

def skin_plot_show(train_x, train_y, w):#��Skin Segmentation���ݼ������н��������άչʾ
    xa = []
    xb = []
    ya = []
    yb = []
    za = []
    zb = []
    w0 = w[0][0]
    w1 = w[0][1]
    w2 = w[0][2]
    w3 = w[0][3]

    for i in range(0, train_x.shape[0]):
        if train_y[i] == 0:
            xa.append(train_x[i][0])
            ya.append(train_x[i][1])
            za.append(train_x[i][2])
        elif train_y[i] == 1:
            xb.append(train_x[i][0])
            yb.append(train_x[i][1])
            zb.append(train_x[i][2])
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xa, ya, za, c='red', s=10, label='class A')
    ax.scatter(xb, yb, zb, c='blue', s=10, label='class B')#����ɢ��ͼ�����ղ�ͬ��ǩ������ɫ
    real_x = np.arange(np.min(train_x[:,0]), np.max(train_x[:,0]), 1)
    real_y = np.arange(np.min(train_x[:,1]), np.max(train_x[:,1]), 1)
    real_X, real_Y = np.meshgrid(real_x, real_y)
    real_z = - w0 / w3 - (w1 / w3) * real_X - (w2 / w3) * real_Y
    ax.plot_surface(real_X, real_Y, real_z)
    ax.legend(loc='best')

    plt.title("Skin Dataset Classification")
    plt.show()

    return 0

def skin_show():#��skin���ݼ���ѵ���������չʾ
    train_x, train_y, test_x, test_y = skin_read()
    X = generate_X(train_x)
    w, count, loss_list = gradient_descent(train_x, train_y)
    test_X = generate_X(test_x)
    acc = accuracy(test_X, test_y, w)
    print(w)
    print("acc:" + str(acc))
    skin_plot_show(train_x, train_y, w)

    return w, count, loss_list

#����������
method = 1
if method == 1:
    skin_show()
elif method == 2:
    gradient_descent_show()
