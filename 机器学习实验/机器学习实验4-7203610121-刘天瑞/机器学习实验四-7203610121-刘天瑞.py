# -*- coding: gbk -*-

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#ʵ��ؼ���������
number = 300#���ɵ�ԭ����������
dimension = 3#ԭ�������ά��
new_dimension = 1#PCA�㷨������������ά��
method = 2#1:���Լ����ɵ����ݽ���PCA�㷨����2:��������ͼ�����ݽ���PCA�㷨����
face_dimension = 100#����ͼ��Ҫ������ά��
filepath = "C:\\face"#��������ͼ��ľ���·��
size = (60, 60)#���òü�Ϊͳһͼ��ߴ��С


def generate_data(number, dimension):
    "��������"
    data = np.zeros((dimension, number))

    if dimension == 2:#��ά���������
        mean = [2, -2]
        cov = [[1, 0], [0, 0.01]]
    elif dimension == 3:#��ά���������
        mean = [2, 2, 3]
        cov = [[1, 0, 0], [0, 1, 0], [0, 0, 0.01]]

    data = np.random.multivariate_normal(mean, cov, number)#���ö�ά��˹�ֲ�����������
    #print(data.shape)#(100,2)

    return data


def PCA(data, new_dimension):
    "ʵ��PCA�㷨"
    x_mean = np.sum(data, axis = 0) / number#��ȡԭ���ݾ�ֵ
    decentral_data = data - x_mean#��ɢ���������Ļ�����
    cov = decentral_data.T @ decentral_data#�������Ļ����ݵ�Э�������
    eigenvalues, eigenvectors = np.linalg.eig(cov)#��ȡ����ֵ�������������ٽ��зֽ�
    eigenvectors = np.real(eigenvectors)#ȡʵ��ֵ��ȥ�鲿����
    dimension_order = np.argsort(eigenvalues)#���մ�С�����������������ֵ������
    PCA_vector = eigenvectors[:, dimension_order[:-(new_dimension + 1):-1]]#ѡȡ��������ֵ��Ӧ����������
    x_pca = decentral_data @ PCA_vector @ PCA_vector.T + x_mean#���㾭��PCA�㷨����֮���xֵ

    return PCA_vector, x_mean, x_pca#���������������󣬽�άǰ���ݾ�ֵ�Լ����Ļ�����


def PCA_show():
    "���ӻ�PCA�㷨���"
    data = generate_data(number, dimension)#��������
    _, _, x_pca = PCA(data, new_dimension)#ִ��PCA�㷨

    #����ɢ��ͼ
    if dimension == 2:#ԭ������ά��Ϊ2
        plt.scatter(data.T[0], data.T[1], c='w',
                    edgecolors='#008000', s=20, marker='o', label='original data')#ԭʼͼ��ɢ��Ϊ����ɫ
        plt.scatter(x_pca.T[0], x_pca.T[1], c='w',
                    edgecolors='#FFD700', s=20, marker='o', label='PCA data')#PCA��ͼ��ɢ��Ϊ��ɫ

    elif dimension == 3:#ԭ������ά��Ϊ3
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(data.T[0], data.T[1], data.T[2],
                   c='#008000', s=20, label='original data')#ԭʼͼ��ɢ��Ϊ����ɫ
        ax.scatter(x_pca.T[0], x_pca.T[1], x_pca.T[2],
                   c='#FFD700', s=20, label='PCA data')#PCA��ͼ��ɢ��Ϊ��ɫ

    plt.title("number = %d, dimension = %d, new dimension = %d" % (number, dimension, new_dimension))
    plt.legend()#����ͼ����ǩ˵��
    plt.show()#չʾɢ��ͼ

    return


def read_face_data():
    "�����������ݲ��ҽ���չʾ"
    img_list = os.listdir(filepath)#��ȡͼ���ļ������б�
    data = []
    i = 1#��ʼ��ÿ�еĵ�i��ͼ��

    for img in img_list:
        path = os.path.join(filepath, img)#����ͼ��
        plt.subplot(3, 3, i)#ֱ��ָ�����ַ�ʽ��λ�ý��л�ͼ������3*3��ͼ��
        with open(path) as _:
            img_data = cv2.imread(path)#��ȡͼ��
            img_data = cv2.resize(img_data, size)#ѹ��ͼ����size��С
            img_gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)#RGB��ͨ��ͼ��ת��Ϊ�Ҷ�ͼ
            plt.imshow(img_gray)#����ͼ��Ԥ��
            h, w = img_gray.shape#���ûҶ�ͼ�ĸ߶ȺͿ������
            img_col = img_gray.reshape(h * w)#��(h,w)��ͼ�����ݽ�����ƽ
            data.append(img_col)
        i += 1#˳�ε���
    plt.show()#ͼ��չʾ

    return np.array(data)#(9, 1600)


def PSNR(img_1, img_2):
    "����ͼ�������"
    diff = (img_1 - img_2) ** 2#���ж���������
    mse = np.sqrt(np.mean(diff))#��ȡ��ֵ�Ŀ���

    return 20 * np.log10(255.0 / mse)#PSNR���㹫ʽ


def face_show():
    "��������PCA"
    data = read_face_data()
    n, _ = data.shape
    _, _, x_pca = PCA(data, face_dimension)
    x_pca = np.real(x_pca)#ȡʵ��ֵ��ȥ�鲿����

    # ����PCA�㷨������ͼ��
    plt.figure()

    for i in range(n):
        plt.subplot(3, 3, i+1)
        plt.imshow(x_pca[i].reshape(size))
    plt.show()

    # ���������
    print("ѹ�����ά��Ϊ��%d������������и�����ʾ��" % face_dimension)

    for i in range(n):
        psnr = PSNR(data[i], x_pca[i])
        print("ͼ%d�������Ϊ: %.3f" % (i + 1, psnr))

    return


#ѡ��ʵ�鷽��
if method == 1:
    PCA_show()
elif method == 2:
    face_show()