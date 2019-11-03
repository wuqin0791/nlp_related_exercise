'''
@Description: This is a python file
@Author: JeanneWu
@Date: 2019-10-31 11:29:31
'''
# 男神老吴SVD处理
 
from skimage import io
import matplotlib.pyplot as plt
from numpy import *
from numpy import linalg as la
 
path = 'WechatIMG1316.jpeg'
data = io.imread(path)
data = mat(data)        # 需要mat处理后才能在降维中使用矩阵的相乘
U,sigma,VT = linalg.svd(data)
# 在重构之前，依据前面的方法需要选择达到某个能量度的奇异值
cnt = sum(sigma)
print(cnt)
cnt90 = 0.9*cnt    # 达到90%时的奇异总值
print(cnt90)
count = 50        # 选择前50个奇异值
cntN = sum(sigma[:count])
print(cntN)
 
# 重构矩阵
dig = mat(eye(count)*sigma[:count]) # 获得对角矩阵
# dim = data.T * U[:,:count] * dig.I # 降维 格外变量这里没有用
redata = U[:,:count] * dig * VT[:count,:]   # 重构
 
plt.imshow(redata,cmap='gray')  # 取灰
plt.show()  # 可以使用save函数来保存图片
