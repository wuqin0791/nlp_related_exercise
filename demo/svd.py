'''
@Description: This is a python file
@Author: JeanneWu
@Date: 2019-10-31 10:50:44
'''

from numpy import *
from numpy import linalg as la

# 欧几里德距离 这里返回结果已处理 0，1   0最大相似，1最小相似   欧氏距离转换为2范数计算
def ecludSim(inA,inB):
    return 1.0 / (1.0 + la.norm(inA-inB))
 
# 皮尔逊相关系数 numpy的corrcoef函数计算
def pearsSim(inA,inB):
    if(len(inA) < 3):
        return 1.0
    return 0.5 + 0.5*corrcoef(inA,inB,rowvar=0)[0][1]   # 使用0.5+0.5*x 将-1，1 转为 0，1
 
# 余玄相似度 根据公式带入即可，其中分母为2范数计算，linalg的norm可计算范数
def cosSim(inA,inB):
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5*(num/denom)    # 同样操作转换 0，1

# （用户x商品）    # 为0表示该用户未评价此商品，即可以作为推荐商品
def loadExData2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]
 
# 替代上面的standEst(功能) 该函数用SVD降维后的矩阵来计算评分
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    U,Sigma,VT = la.svd(dataMat)
    Sig4 = mat(eye(4)*Sigma[:4])  #将奇异值向量转换为奇异值矩阵
    # (dataMat.T) => 矩阵的转置
    print(U)
    # U[:,:4]所有行取前四列
    xformedItems = dataMat.T * U[:,:4] * Sig4.I  # 降维方法 通过U矩阵将物品转换到低维空间中 （商品数行x选用奇异值列）
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j == item:
            continue
        # 这里需要说明：由于降维后的矩阵与原矩阵代表数据不同（行由用户变为了商品），所以在比较两件商品时应当取【该行所有列】 再转置为列向量传参
        similarity = simMeas(xformedItems[item,:].T,xformedItems[j,:].T)
        # print('%d 和 %d 的相似度是: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal
 
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=svdEst):
    unratedItems = nonzero(dataMat[user,:].A==0)[1]  # 找到该行所有为0的位置（即此用户未评价的商品，才做推荐）
    if len(unratedItems) == 0:
        return '所有物品都已评价...'
    itemScores = []
    for item in unratedItems:   # 循环所有没有评价的商品列下标
        estimatedScore = estMethod(dataMat, user, simMeas, item)    # 计算当前产品的评分
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]   # 将推荐商品排序

# 结果测试如下：
myMat = mat(loadExData2())
result1 = recommend(myMat,1,estMethod=svdEst)   # 需要传参改变默认函数
print(result1)
result2 = recommend(myMat,1,estMethod=svdEst,simMeas=pearsSim)
