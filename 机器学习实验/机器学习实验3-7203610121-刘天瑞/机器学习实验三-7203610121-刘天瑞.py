# -*- coding: gbk -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd

K_means_epsilon = 1e-7#K-means算法的迭代误差
condition = 1#选择的设置:config_0：0；config_1：1
method = 1#选择的方法：0：K-means；1：UCI
colors = ['#86C166', '#51A8DD', '#B481BB', '#F596AA', '#F7D94C', '#D75455']#提供颜色库

#初始化设置
config_0 = {
    'k': 3,#聚类数
    'n': 400,#每类的样本点数
    'dim': 2,#样本点的维度
    'mu': np.array([[-5, 4], [5, 4], [3, -4], [-5, -5]]),#均值
    'sigma': np.array([[[2, 0], [0, 1]], [[2, 0], [0, 2]], [[3, 0], [0, 1]], [[3, 0], [0, 2]]])#方差
}

config_1 = {
    'k': 6,#聚类数
    'n': 400,#每类的样本点数
    'dim': 2,#样本点的维度
    'mu': np.array([[-4, 3], [4, 2], [1, -4], [-5, -3], [0, 0], [6, -1], [-1, 8], [7, -4]]),#均值
    'sigma': np.array([
        [[2, 0], [0, 1]], [[2, 0], [0, 2]], [[3, 0], [0, 1]], [[3, 0], [0, 2]],
        [[2, 0], [0, 1]], [[2, 0], [0, 2]], [[3, 0], [0, 1]], [[3, 0], [0, 2]]
    ])#方差
}

configs = [config_0, config_1]

def generate_data(k, n, dim, mu, sigma):
    "生成数据"
    raw_data = np.zeros((k, n, dim))
    for i in range(k):
        raw_data[i] = np.random.multivariate_normal(mu[i], sigma[i], n)
    data = np.zeros((k * n, dim))
    for i in range(k):
        data[i * n:(i + 1) * n] = raw_data[i]
    return data


def K_means(data, k, N, dim):
    "实现K-means算法"
    category = np.zeros((N, dim + 1))
    category[:, 0:dim] = data[:, :]#多出的一维矩阵用来保存类别标签
    center = np.zeros((k, dim))#k行dim列，用来保存各类中心
    for i in range(k):#随机以某些样本点作为初始点
        center[i, :] = data[np.random.randint(0, N), :]
    iter_count = 0#迭代次数
    #K-means算法的核心部分
    while True:
        iter_count += 1
        distance = np.zeros(k)#用来保存某次迭代中样本点到所有聚类中心的距离
        for i in range(N):
            point = data[i, :] #选取用来计算距离的样本点
            for j in range(k):
                t_center = center[j, :]#选取用来计算距离的聚类中心
                distance[j] = np.linalg.norm(point - t_center)#更新该样本点到聚类中心的距离
            min_index = np.argmin(distance)#找到该样本点对应距离最近的聚类中心
            category[i, dim] = min_index
        num = np.zeros(k)#保存每个类别的样本点数
        count = 0#计算更新后的距离小于误差值的聚类中心个数
        new_center_sum = np.zeros((k, dim))#临时变量
        new_center = np.zeros((k, dim))#保存此次迭代得到的新聚类中心
        for i in range(N):
            label = int(category[i, dim])
            num[label] += 1#统计各类别的样本点数
            new_center_sum[label, :] += category[i, :dim]
        for i in range(k):
            if num[i] != 0:
                new_center[i, :] = new_center_sum[i, :] / \
                    num[i]#计算本次样本点所得到的聚类中心
            new_k_distance = np.linalg.norm(new_center[i, :] - center[i, :])
            if new_k_distance < K_means_epsilon:#计算更新后的距离小于误差值的聚类中心个数
                count += 1
        if count == k:#当所有聚类中心更新后的最小距离都小于误差值时，结束循环
            return category, center, iter_count, num
        else:#否则更新聚类中心
            center = new_center


def calculate_accuracy(num, n, N):
    "计算K-means算法的分类准确率"
    """主要思想：首先将各类样本点数都减去最开始分类时每一类别的总数目，再求其中元素的绝对值之和，
    然后除以2，最后得到的即为分类错误点的个数"""
    num_1 = np.abs(num - n)
    error = np.sum(num_1) / 2
    return 1 - error / N


def K_means_show(k, n, dim, mu, sigma):
    "K-means算法的结果展示"
    data = generate_data(k, n, dim, mu, sigma)
    #print(data.shape)#(1200,2)
    N = data.shape[0]#N为样本点总数
    category, center, iter_count, num = K_means(data, k, N, dim)
    accuracy = calculate_accuracy(num, n, N)
    for i in range(N):#绘制已分类的样本点
        color_num = int(category[i, dim] % len(colors))
        plt.scatter(category[i, 0], category[i, 1],
                    c=colors[color_num], s=5)
    for i in range(k):
        plt.scatter(center[i, 0], center[i, 1],
                    c='red', marker='x')#绘制所有的聚类中心
    print("accuracy:" + str(accuracy))
    plt.title("K-means results " + "   config:" +
              str(condition) + "   iter:" + str(iter_count))
    plt.show()
    return


def UCI_read():
    "读入UCI数据并进行切分"
    raw_data = pd.read_csv("./iris.csv")
    data = raw_data.values
    label = data[:, -1]
    np.delete(data, -1, axis = 1)
    #print(data.shape)#(150, 5)
    return data, label


def UCI_show():
    "使用UCI数据集测试K-means算法"
    data, _ = UCI_read()
    k = 6#聚类数
    N = data.shape[0]#样本数量
    n = N / k#每一类样本数量
    dim = data.shape[1]#样本维度
    #K-means算法的计算结果
    _, _, iter_count, num = K_means(data, k, N, dim)
    K_means_accuracy = calculate_accuracy(num, n, N)
    print("K-means算法结果：")
    print("迭代次数：%d   分类准确率:%.2f\n" % (iter_count, K_means_accuracy))
    return


#主函数
config = configs[condition]#选取配置
k, n, dim, mu, sigma = config['k'], config['n'], config['dim'], config['mu'], config['sigma']
if method == 0:
    K_means_show(k, n, dim, mu, sigma)
elif method == 1:
    UCI_show()
