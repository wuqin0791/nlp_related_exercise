'''
@Description: 这是我的代码复现
@Author: JeanneWu
@Date: 2019-11-01 15:12:15
'''

from collections import Counter,defaultdict
from scipy.spatial.distance import cosine
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from icecream import ic

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# 1. Re-code the Linear-Regression Model using scikit-learning(10 points)
def assmuing_function(x): #定义一个随机函数，不然实在太random了
    return random.randint(0,4)
randomMatrix = np.random.random((20, 2)) #生成一个20*2的矩阵
# print(random)
X = randomMatrix[:, 0] #取第一列
Y = randomMatrix[:, 1] #取第二列

y = [assmuing_function(x) for x in X]
y = np.array(y)

# print(y)
# plt.scatter(X, y)
# plt.show()
reg = LinearRegression().fit(X.reshape(-1, 1), y)


reg.score(X.reshape(-1, 1), y)
k = reg.coef_
b = reg.intercept_
print('parameter k is %.3f' % k) #保留三位小数
def prediction(x):
    return k * x + b

plt.scatter(X, y)
plt.plot(X, prediction(X))
# plt.show() #需要的时候再打开




# 2. Complete the unfinished KNN Model using pure python to solve the previous Line-Regression problem. (8 points)
# 是否完成了KNN模型 (4') finish
# 是否能够预测新的数据 (4')
def model(X,y):
 
    return [(Xi, yi) for Xi, yi in zip(X, y)]

def distance(x1, x2):
    return cosine(x1, x2)

def predict(x, k=5):
    arr = []
    # sort = np.argsort( distance(xi[0], x))
    # print(sort)
    most_similars = sorted(model(X, y), key=lambda xi: distance(xi[0], x))[:k]
    for sort_y in most_similars:

        arr.append(sort_y[1])

    votes = Counter(arr)
 
    predict_y = votes.most_common(1)[0][0]
    return predict_y


predict((0.54, 7.5))


# 3. Re-code the Decision Tree, which could sort the features by salience. (12 points)

def entropy(elements): 
    counter = Counter(elements) #counter在这里是用来计数的
    print(counter)
    probs = [counter[c] / len(elements) for c in set(elements)]
    ic(probs)
    return - sum(p * np.log(p) for p in probs)

# print(entropy([1,1,1,10]))

mock_data = {
    'gender':['F', 'F', 'F', 'F', 'M', 'M', 'M'],
    'income': ['+10', '-10', '+10', '+10', '+10', '+10', '-10'],
    'family_number': [1, 1, 2, 1, 1, 1, 2],
    'bought': [1, 1, 1, 0, 0, 0, 1],
}
dataset = pd.DataFrame.from_dict(mock_data)

sub_split_1 = dataset[dataset['gender'] != 'F']['bought'].tolist()
sub_split_2 = dataset[dataset['income'] != 'F']['bought'].tolist()
# 可以传入gender/income/family_number/bought
def sortFeaturesBySalience(index):
    return entropy(dataset[dataset['gender'] != 'F']['bought'].tolist())

(sortFeaturesBySalience("family_number"))

def find_the_optimal_spilter(training_data: pd.DataFrame, target: str) -> str: #python3 新特性 指定返回类型
    x_fields = set(training_data.columns.tolist()) - {target} #除去target进行下面的循环
    print(x_fields)

    spliter = None
    min_entropy = float('inf') #表示正无穷
    
    for f in x_fields:
        ic(f)
        values = set(training_data[f])
        ic(values)
        for v in values:
            sub_spliter_1 = training_data[training_data[f] == v][target].tolist()
            ic(sub_split_1)

            entropy_1 = entropy(sub_spliter_1)
            ic(entropy_1)

            sub_spliter_2 = training_data[training_data[f] != v][target].tolist()
            ic(sub_split_2)

            entropy_2 = entropy(sub_spliter_2)
            ic(entropy_2)

            entropy_v = entropy_1 + entropy_2
            ic(entropy_v)
            
            if entropy_v <= min_entropy:
                min_entropy = entropy_v
                spliter = (f, v)
    
    print('spliter is: {}'.format(spliter))
    print('the min entropy is: {}'.format(min_entropy))
    
    return spliter
# find_the_optimal_spilter(dataset, 'bought')

# 4. Finish the K-Means using 2-D matplotlib (8 points)¶

X = [random.randint(0,100) for _ in range(100)]
Y = [random.randint(0,100) for _ in range(100)]
plt.scatter(X,Y)
# plt.show()

training_data = [[x,y] for x,y in zip(X, Y)]  # [[x1,y1], [x2, y2]]
cluster = KMeans(n_clusters=6, max_iter=500)

cluster.fit(training_data)

cluster.cluster_centers_ #计算出6个中心点

cluster.labels_  #每个点属于的类

centers = defaultdict(list)

for labels_, locations in zip(cluster.labels_, training_data):
    centers[labels_].append(locations)

print(centers)
color = ['red', 'green', 'grey', 'black', 'yellow', 'orange']

for i,c in enumerate(centers):
    for loc in centers[c]:
        plt.scatter(*loc, c=color[i])

for center in cluster.cluster_centers_:
    plt.scatter(*center, s=100)

# plt.show()