# -*- coding: gbk -*-

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#实验关键参数配置
number = 300#生成的原样本点数量
dimension = 3#原样本点的维度
new_dimension = 1#PCA算法处理后样本点的维度
method = 2#1:用自己生成的数据进行PCA算法处理；2:利用人脸图像数据进行PCA算法处理
face_dimension = 100#人脸图像要降至的维度
filepath = "C:\\face"#人脸数据图像的具体路径
size = (60, 60)#设置裁剪为统一图像尺寸大小


def generate_data(number, dimension):
    "生成数据"
    data = np.zeros((dimension, number))

    if dimension == 2:#二维数据情况下
        mean = [2, -2]
        cov = [[1, 0], [0, 0.01]]
    elif dimension == 3:#三维数据情况下
        mean = [2, 2, 3]
        cov = [[1, 0, 0], [0, 1, 0], [0, 0, 0.01]]

    data = np.random.multivariate_normal(mean, cov, number)#利用多维高斯分布生成样本点
    #print(data.shape)#(100,2)

    return data


def PCA(data, new_dimension):
    "实现PCA算法"
    x_mean = np.sum(data, axis = 0) / number#求取原数据均值
    decentral_data = data - x_mean#将散点数据中心化处理
    cov = decentral_data.T @ decentral_data#计算中心化数据的协方差矩阵
    eigenvalues, eigenvectors = np.linalg.eig(cov)#求取特征值和特征向量，再进行分解
    eigenvectors = np.real(eigenvectors)#取实部值（去虚部处理）
    dimension_order = np.argsort(eigenvalues)#按照从小到大排序来获得特征值的索引
    PCA_vector = eigenvectors[:, dimension_order[:-(new_dimension + 1):-1]]#选取最大的特征值对应的特征向量
    x_pca = decentral_data @ PCA_vector @ PCA_vector.T + x_mean#计算经过PCA算法分析之后的x值

    return PCA_vector, x_mean, x_pca#返回特征向量矩阵，降维前数据均值以及中心化数据


def PCA_show():
    "可视化PCA算法结果"
    data = generate_data(number, dimension)#生成数据
    _, _, x_pca = PCA(data, new_dimension)#执行PCA算法

    #绘制散点图
    if dimension == 2:#原样本点维数为2
        plt.scatter(data.T[0], data.T[1], c='w',
                    edgecolors='#008000', s=20, marker='o', label='original data')#原始图像散点为深绿色
        plt.scatter(x_pca.T[0], x_pca.T[1], c='w',
                    edgecolors='#FFD700', s=20, marker='o', label='PCA data')#PCA后图像散点为金色

    elif dimension == 3:#原样本点维数为3
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(data.T[0], data.T[1], data.T[2],
                   c='#008000', s=20, label='original data')#原始图像散点为深绿色
        ax.scatter(x_pca.T[0], x_pca.T[1], x_pca.T[2],
                   c='#FFD700', s=20, label='PCA data')#PCA后图像散点为金色

    plt.title("number = %d, dimension = %d, new dimension = %d" % (number, dimension, new_dimension))
    plt.legend()#给出图例标签说明
    plt.show()#展示散点图

    return


def read_face_data():
    "读入人脸数据并且进行展示"
    img_list = os.listdir(filepath)#获取图像文件夹名列表
    data = []
    i = 1#初始化每行的第i个图像

    for img in img_list:
        path = os.path.join(filepath, img)#输入图像
        plt.subplot(3, 3, i)#直接指定划分方式和位置进行绘图，共有3*3张图像
        with open(path) as _:
            img_data = cv2.imread(path)#读取图像
            img_data = cv2.resize(img_data, size)#压缩图像至size大小
            img_gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)#RGB三通道图像转换为灰度图
            plt.imshow(img_gray)#进行图像预览
            h, w = img_gray.shape#设置灰度图的高度和宽度数据
            img_col = img_gray.reshape(h * w)#对(h,w)的图像数据进行拉平
            data.append(img_col)
        i += 1#顺次迭代
    plt.show()#图像展示

    return np.array(data)#(9, 1600)


def PSNR(img_1, img_2):
    "计算图像信噪比"
    diff = (img_1 - img_2) ** 2#进行二阶幂运算
    mse = np.sqrt(np.mean(diff))#求取均值的开方

    return 20 * np.log10(255.0 / mse)#PSNR计算公式


def face_show():
    "人脸数据PCA"
    data = read_face_data()
    n, _ = data.shape
    _, _, x_pca = PCA(data, face_dimension)
    x_pca = np.real(x_pca)#取实部值（去虚部处理）

    # 绘制PCA算法处理后的图像
    plt.figure()

    for i in range(n):
        plt.subplot(3, 3, i+1)
        plt.imshow(x_pca[i].reshape(size))
    plt.show()

    # 计算信噪比
    print("压缩后的维度为：%d，信噪比如下列各行所示：" % face_dimension)

    for i in range(n):
        psnr = PSNR(data[i], x_pca[i])
        print("图%d，信噪比为: %.3f" % (i + 1, psnr))

    return


#选择实验方案
if method == 1:
    PCA_show()
elif method == 2:
    face_show()