'''
@Description: 测试for循环和matrix的计算时间，并进行比较
@Author: JeanneWu
@Date: 2019-11-16 21:38:08
'''
import time as tm
import numpy as np
import collections
import math 

count = 1000000  #一百万数据

x1 = np.ones(count)
y1 = np.ones(count)

r_interation = collections.defaultdict(list)
inter_start = tm.clock()
# print(tm.time())
for i in range(count):
    r_interation[i] = x1[i] * y1[i]

inter_stop = tm.clock()
differ_inter = inter_stop - inter_start
print('这是for循环所用的时间：',differ_inter)

matrix_start = tm.clock()
r_matrix = x1 * y1
matrix_stop = tm.clock()
differ_matrix = matrix_stop - matrix_start
print("这是矩阵计算所花的时间：",differ_matrix)

print("for循环比matrix计算慢",math.floor(differ_inter/differ_matrix),"倍")

