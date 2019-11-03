'''
@Description: This is a python file
@Author: JeanneWu
@Date: 2019-11-02 17:24:03
'''
from scipy.spatial.distance import cosine
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from icecream import ic
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

def classify_two(inX, dataSet, labels, k):
    m, n = dataSet.shape   # shape（m, n）m列n个特征
    # 计算测试数据到每个点的欧式距离
    distances = []
    for i in range(m):
        sum = 0
        for j in range(n):
            sum += (inX[j] - dataSet[i][j]) ** 2
        distances.append(sum ** 0.5)

    sortDist = sorted(distances)
 
    print(sortDist)
    # k 个最近的值所属的类别
    classCount = {}
    for i in range(k):
        voteLabel = labels[distances.index(sortDist[i])]
        print(distances.index(sortDist[i]))
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1 # 0:map default
        print(classCount[voteLabel])
    sortedClass = sorted(classCount.items(), key=lambda d:d[1], reverse=True)
    return sortedClass[0][0]


def createDataSet():
    group = np.array([[1, 1.1], [1, 1], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

if __name__ == '__main__':
    dataSet, labels = createDataSet()
    r = classify_two([0, 0.2], dataSet, labels, 3)
    print(r)
