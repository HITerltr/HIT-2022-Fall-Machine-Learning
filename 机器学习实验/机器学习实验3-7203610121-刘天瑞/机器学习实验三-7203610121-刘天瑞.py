# -*- coding: gbk -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd

K_means_epsilon = 1e-7#K-means�㷨�ĵ������
condition = 1#ѡ�������:config_0��0��config_1��1
method = 1#ѡ��ķ�����0��K-means��1��UCI
colors = ['#86C166', '#51A8DD', '#B481BB', '#F596AA', '#F7D94C', '#D75455']#�ṩ��ɫ��

#��ʼ������
config_0 = {
    'k': 3,#������
    'n': 400,#ÿ�����������
    'dim': 2,#�������ά��
    'mu': np.array([[-5, 4], [5, 4], [3, -4], [-5, -5]]),#��ֵ
    'sigma': np.array([[[2, 0], [0, 1]], [[2, 0], [0, 2]], [[3, 0], [0, 1]], [[3, 0], [0, 2]]])#����
}

config_1 = {
    'k': 6,#������
    'n': 400,#ÿ�����������
    'dim': 2,#�������ά��
    'mu': np.array([[-4, 3], [4, 2], [1, -4], [-5, -3], [0, 0], [6, -1], [-1, 8], [7, -4]]),#��ֵ
    'sigma': np.array([
        [[2, 0], [0, 1]], [[2, 0], [0, 2]], [[3, 0], [0, 1]], [[3, 0], [0, 2]],
        [[2, 0], [0, 1]], [[2, 0], [0, 2]], [[3, 0], [0, 1]], [[3, 0], [0, 2]]
    ])#����
}

configs = [config_0, config_1]

def generate_data(k, n, dim, mu, sigma):
    "��������"
    raw_data = np.zeros((k, n, dim))
    for i in range(k):
        raw_data[i] = np.random.multivariate_normal(mu[i], sigma[i], n)
    data = np.zeros((k * n, dim))
    for i in range(k):
        data[i * n:(i + 1) * n] = raw_data[i]
    return data


def K_means(data, k, N, dim):
    "ʵ��K-means�㷨"
    category = np.zeros((N, dim + 1))
    category[:, 0:dim] = data[:, :]#�����һά����������������ǩ
    center = np.zeros((k, dim))#k��dim�У����������������
    for i in range(k):#�����ĳЩ��������Ϊ��ʼ��
        center[i, :] = data[np.random.randint(0, N), :]
    iter_count = 0#��������
    #K-means�㷨�ĺ��Ĳ���
    while True:
        iter_count += 1
        distance = np.zeros(k)#��������ĳ�ε����������㵽���о������ĵľ���
        for i in range(N):
            point = data[i, :] #ѡȡ������������������
            for j in range(k):
                t_center = center[j, :]#ѡȡ�����������ľ�������
                distance[j] = np.linalg.norm(point - t_center)#���¸������㵽�������ĵľ���
            min_index = np.argmin(distance)#�ҵ����������Ӧ��������ľ�������
            category[i, dim] = min_index
        num = np.zeros(k)#����ÿ��������������
        count = 0#������º�ľ���С�����ֵ�ľ������ĸ���
        new_center_sum = np.zeros((k, dim))#��ʱ����
        new_center = np.zeros((k, dim))#����˴ε����õ����¾�������
        for i in range(N):
            label = int(category[i, dim])
            num[label] += 1#ͳ�Ƹ�������������
            new_center_sum[label, :] += category[i, :dim]
        for i in range(k):
            if num[i] != 0:
                new_center[i, :] = new_center_sum[i, :] / \
                    num[i]#���㱾�����������õ��ľ�������
            new_k_distance = np.linalg.norm(new_center[i, :] - center[i, :])
            if new_k_distance < K_means_epsilon:#������º�ľ���С�����ֵ�ľ������ĸ���
                count += 1
        if count == k:#�����о������ĸ��º����С���붼С�����ֵʱ������ѭ��
            return category, center, iter_count, num
        else:#������¾�������
            center = new_center


def calculate_accuracy(num, n, N):
    "����K-means�㷨�ķ���׼ȷ��"
    """��Ҫ˼�룺���Ƚ�����������������ȥ�ʼ����ʱÿһ��������Ŀ����������Ԫ�صľ���ֵ֮�ͣ�
    Ȼ�����2�����õ��ļ�Ϊ��������ĸ���"""
    num_1 = np.abs(num - n)
    error = np.sum(num_1) / 2
    return 1 - error / N


def K_means_show(k, n, dim, mu, sigma):
    "K-means�㷨�Ľ��չʾ"
    data = generate_data(k, n, dim, mu, sigma)
    #print(data.shape)#(1200,2)
    N = data.shape[0]#NΪ����������
    category, center, iter_count, num = K_means(data, k, N, dim)
    accuracy = calculate_accuracy(num, n, N)
    for i in range(N):#�����ѷ����������
        color_num = int(category[i, dim] % len(colors))
        plt.scatter(category[i, 0], category[i, 1],
                    c=colors[color_num], s=5)
    for i in range(k):
        plt.scatter(center[i, 0], center[i, 1],
                    c='red', marker='x')#�������еľ�������
    print("accuracy:" + str(accuracy))
    plt.title("K-means results " + "   config:" +
              str(condition) + "   iter:" + str(iter_count))
    plt.show()
    return


def UCI_read():
    "����UCI���ݲ������з�"
    raw_data = pd.read_csv("./iris.csv")
    data = raw_data.values
    label = data[:, -1]
    np.delete(data, -1, axis = 1)
    #print(data.shape)#(150, 5)
    return data, label


def UCI_show():
    "ʹ��UCI���ݼ�����K-means�㷨"
    data, _ = UCI_read()
    k = 6#������
    N = data.shape[0]#��������
    n = N / k#ÿһ����������
    dim = data.shape[1]#����ά��
    #K-means�㷨�ļ�����
    _, _, iter_count, num = K_means(data, k, N, dim)
    K_means_accuracy = calculate_accuracy(num, n, N)
    print("K-means�㷨�����")
    print("����������%d   ����׼ȷ��:%.2f\n" % (iter_count, K_means_accuracy))
    return


#������
config = configs[condition]#ѡȡ����
k, n, dim, mu, sigma = config['k'], config['n'], config['dim'], config['mu'], config['sigma']
if method == 0:
    K_means_show(k, n, dim, mu, sigma)
elif method == 1:
    UCI_show()
