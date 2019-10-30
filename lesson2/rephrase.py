'''
@Description: 这是课堂代码复现
@Author: JeanneWu
@Date: 2019-10-19 15:10:50
'''
coordination_source = """
{name:'兰州', geoCoord:[103.73, 36.03]},
{name:'嘉峪关', geoCoord:[98.17, 39.47]},
{name:'西宁', geoCoord:[101.74, 36.56]},
{name:'成都', geoCoord:[104.06, 30.67]},
{name:'石家庄', geoCoord:[114.48, 38.03]},
{name:'拉萨', geoCoord:[102.73, 25.04]},
{name:'贵阳', geoCoord:[106.71, 26.57]},
{name:'武汉', geoCoord:[114.31, 30.52]},
{name:'郑州', geoCoord:[113.65, 34.76]},
{name:'济南', geoCoord:[117, 36.65]},
{name:'南京', geoCoord:[118.78, 32.04]},
{name:'合肥', geoCoord:[117.27, 31.86]},
{name:'杭州', geoCoord:[120.19, 30.26]},
{name:'南昌', geoCoord:[115.89, 28.68]},
{name:'福州', geoCoord:[119.3, 26.08]},
{name:'广州', geoCoord:[113.23, 23.16]},
{name:'长沙', geoCoord:[113, 28.21]},
//{name:'海口', geoCoord:[110.35, 20.02]},
{name:'沈阳', geoCoord:[123.38, 41.8]},
{name:'长春', geoCoord:[125.35, 43.88]},
{name:'哈尔滨', geoCoord:[126.63, 45.75]},
{name:'太原', geoCoord:[112.53, 37.87]},
{name:'西安', geoCoord:[108.95, 34.27]},
//{name:'台湾', geoCoord:[121.30, 25.03]},
{name:'北京', geoCoord:[116.46, 39.92]},
{name:'上海', geoCoord:[121.48, 31.22]},
{name:'重庆', geoCoord:[106.54, 29.59]},
{name:'天津', geoCoord:[117.2, 39.13]},
{name:'呼和浩特', geoCoord:[111.65, 40.82]},
{name:'南宁', geoCoord:[108.33, 22.84]},
//{name:'西藏', geoCoord:[91.11, 29.97]},
{name:'银川', geoCoord:[106.27, 38.47]},
{name:'乌鲁木齐', geoCoord:[87.68, 43.77]},
{name:'香港', geoCoord:[114.17, 22.28]},
{name:'澳门', geoCoord:[113.54, 22.19]}
"""

import re
import math
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


# 正则表达式的练习 *有或者没有 +有一个或者更多
# pattern = re.compile('colou?r')
# c = pattern.findall('color or colour?')
# print(c)
# exit()

def get_city_info(city_coordination):
    city_location = {}
    for line in city_coordination.split("\n"):
        if line.startswith("//"): continue
        if line.strip() == "":continue
            
        city = re.findall("name:'(\w+)'",line)[0] #因为提取出来的时候，是个数组，要取第零项

        x_y = re.findall("Coord:\[(\d+.\d+),\s(\d+.\d+)\]",line)[0]
        # print(x_y,'x_y')
        # exit()
        x_y = tuple(map(float,x_y))
        # print(map(float,x_y))
        city_location[city] = x_y
    return city_location

city_info = get_city_info(coordination_source)


#计算两点之间的距离 外部拷贝

'''
@description: 
@param {经度，纬度} 
@return: 
'''
def geo_distance(origin, destination):
    """
    Calculate the Haversine distance.

    Parameters
    ----------
    origin : tuple of float
        (lat, long)
    destination : tuple of float
        (lat, long)

    Returns
    -------
    distance_in_km : float

    Examples
    --------
    >>> origin = (48.1372, 11.5756)  # Munich
    >>> destination = (52.5186, 13.4083)  # Berlin
    >>> round(distance(origin, destination), 1)
    504.2
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d

def get_city_distance(city1,city2):
    return geo_distance(city_info[city1],city_info[city2])

# print(get_city_distance("北京","上海"))

# print(city_info.keys())

city_graph = nx.Graph()  #建一张图

city_graph.add_nodes_from(list(city_info.keys()))

# %matplotlib inline

(nx.draw(city_graph, city_info, with_labels=True, node_size=10))

threshold = 700

def build_connection(city_info):
    cities_connection = defaultdict(list)
    cities = list(city_info.keys())
    for c1 in cities:
        for c2 in cities:
            if c1 == c2 : continue
            
            if get_city_distance(c1,c2) < threshold:
                cities_connection[c1].append(c2)
    return cities_connection

cities_connection = build_connection(city_info)
print(cities_connection)

cities_connection_graph = (nx.Graph(cities_connection))
nx.draw(cities_connection_graph,city_info,with_labels=True,node_size=10)
print(city_info)

plt.show()
exit()

#接下来是BFC的算法实现
def bfc(graph, start, destination): 
    pathes = [[start]]
    
    visited = set()

    while pathes: 
        # print(pathes,11)
        path = pathes.pop(0) #最开始的点
        frontier = path[-1] #最尾巴的点

        # print(path)
        if frontier in visited: continue #去重

        successsors = graph[frontier]
        for city in successsors:
            if city in path: continue #检查环
            new_path = path + [city]
            # print(path)
       
            pathes.append(new_path) #核心在这里
            # print(pathes)
            if city == destination:  #如果是最后一个路径的话，就return 返回
                return new_path
            
            visited.add(frontier)

bfc(cities_connection,"上海","香港")

#  todo
def bfc_2(graph, start, destination, search_strategy):
    pathes = [[start]]

    while pathes:
        path = pathes.pop(0)
        frontier = path[-1]

        successsors = graph[frontier]
        for city in successsors:
            if city in path: continue

            new_path = path + [city]
            pathes.append(new_path)

    pathes = search_strategy(pathes)

    if pathes and pathes[0][1] == destination:
        return pathes[0]

def sort_by_distance(pathes):
    def get_distance_of_path(path):
        distance = 0 
        for i,_ in enumerate(path[:-1]):
            distance += get_city_distance(path[i], path[i+1])
        return distance
    return sorted(pathes, key=get_distance_of_path)

# 机器学习 梯度下降部分
from sklearn.datasets import load_boston #sklearn一个机器学习的库
import random

dataset = load_boston()   
x,y=dataset['data'],dataset['target'] 

X_rm = x[:,5] # 以RM为例

#define target function
def price(rm, k, b):
    return k * rm + b

# define loss function 
def loss(y,y_hat):
    return sum((y_i - y_hat_i)**2 for y_i, y_hat_i in zip(list(y),list(y_hat)))/len(list(y))

# define partial derivative 
def partial_derivative_k(x, y, y_hat):
    n = len(y)
    gradient = 0
    for x_i, y_i, y_hat_i in zip(list(x),list(y),list(y_hat)):
        gradient += (y_i-y_hat_i) * x_i
    return -2/n * gradient

def partial_derivative_b(y, y_hat):
    n = len(y)
    gradient = 0
    for y_i, y_hat_i in zip(list(y),list(y_hat)):
        gradient += (y_i-y_hat_i)
    return -2 / n * gradient

k = random.random() * 200 - 100  # -100 100
b = random.random() * 200 - 100  # -100 100

learning_rate = 1e-3

iteration_num = 200 
losses = []
for i in range(iteration_num):
    
    price_use_current_parameters = [price(r, k, b) for r in X_rm]  # \hat{y}
    
    current_loss = loss(y, price_use_current_parameters)
    losses.append(current_loss)
    print("Iteration {}, the loss is {}, parameters k is {} and b is {}".format(i,current_loss,k,b))
    
    k_gradient = partial_derivative_k(X_rm, y, price_use_current_parameters)
    b_gradient = partial_derivative_b(y, price_use_current_parameters)
    
    k = k + (-1 * k_gradient) * learning_rate
    b = b + (-1 * b_gradient) * learning_rate
best_k = k
best_b = b
    