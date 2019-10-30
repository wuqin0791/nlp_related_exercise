'''
@Description: 这是我爬取的深圳地铁的数据，然后用BFS实现搜索
@Author: JeanneWu
@Date: 2019-10-29 11:01:38
'''
"""
@Description: This is a python file
@Author: JeanneWu
@Date: 2019-10-29 11:01:38
"""
from collections import defaultdict
import networkx as nx
import requests
from bs4 import BeautifulSoup
import re
import matplotlib.pyplot as plt


def targetFun(tag):
    return tag.has_attr('href')
response = requests.get('http://jtapi.bendibao.com/ditie_mip/metro.aspx?citycode=sz')

soup = BeautifulSoup(response.text, 'html.parser')

k = 0
b = 0
arr = dict()
for item in soup.find_all('ul', 'station-li'):

    temp = item.find_all('li')
    arr[b] = []
    for i in temp:
        arr[b].append(i.a.string)
        k += 1
    b += 1


def geocodeG(address):
    # if address == '老街': return "114.116939,22.544232"
    par = {
        "address": "广东省深圳市" + address + "地铁站",
        "key": "928ba2b8a37823e13884ff6653072c86",
    }
    base = "http://restapi.amap.com/v3/geocode/geo"
    response = requests.get(base, par)
    answer = response.json()
    # print(answer)
    GPS=answer['geocodes'][0]['location'].split(",")
    # print(GPS[0],111)
    # exit()
    # print(map(float,(GPS[0], GPS[1])))
    return tuple(map(float,(GPS[0], GPS[1])))


arr2 = []
cities_connection = defaultdict(list)
city_info = dict()
for item in arr.values():
    frontier = item[-1]
    count = 0
    for i in item:
        print(i)
        print(geocodeG(i))
        city_info[i] = geocodeG(i)
        if i == frontier:
            cities_connection[i] = [item[count - 1]]
            continue
        cities_connection[i] = [item[count + 1]]
        count += 1

# print(cities_connection)

cities_connection_graph = nx.Graph(cities_connection)
nx.draw(cities_connection_graph, city_info, with_labels=True, node_size=10)
plt.show()

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
        # print(successsors)
        for city in successsors:
            if city in path: continue #检查环
            new_path = path + [city]
            # print(path)
       
            pathes.append(new_path) #核心在这里
            # print(pathes)
            if city == destination:  #如果是最后一个路径的话，就return 返回
                return new_path
            
            visited.add(frontier)

print(bfc(cities_connection,"机场东","后瑞"))



