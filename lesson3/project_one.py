'''
@Description: 这是我的编程题第一题的实现
@Author: JeanneWu
@Date: 2019-11-03 06:36:19
'''
from collections import Counter
from icecream import ic
import numpy as np
import pandas as pd



def entropy(elements): 
    counter = Counter(elements) #counter在这里是用来计数的
    print(counter)
    probs = [counter[c] / len(elements) for c in set(elements)]
    # ic(probs)
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
# def sortFeaturesBySalience(index):
#     return entropy(dataset[dataset['gender'] != 'F']['bought'].tolist())

# print(sortFeaturesBySalience("family_number"))

def find_the_optimal_spilter(training_data: pd.DataFrame, target: str) -> str: #python3 新特性 指定返回类型
    x_fields = set(training_data.columns.tolist()) - {target} #除去target进行下面的循环

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
            
            # if entropy_1 == 0.0 : 
            #     return (target, 1)
            # if entropy_2 == 0.0:
            #     return (target, 0)

            if entropy_v <= min_entropy:
                min_entropy = entropy_v
                spliter = (f, v)
                # datasetNew =  dataset[dataset[f] != v]
                # find_the_optimal_spilter(datasetNew, target)
    
    print('spliter is: {}'.format(spliter))
    print('the min entropy is: {}'.format(min_entropy))
    
    return spliter


def predicate(dictionary,target):
    buy = (target, 1)
    unbuy = (target, 0)
    firstDecision =  find_the_optimal_spilter(dataset, target)
    if dictionary[firstDecision[0]] == firstDecision[1]: #检测是否满足第一个分岔路
        return buy
    else:
        datasetNew = dataset[dataset[firstDecision[0]] != firstDecision[1]] #不满足就进行第二次决策

        secondDecision = find_the_optimal_spilter(datasetNew, target)

        if dictionary[secondDecision[0]] == secondDecision[1]:
            return buy
        else: #否则进行第三次决策
            datasetThird = dataset[dataset[secondDecision[0]] != secondDecision[1]]
            thirdDecision = find_the_optimal_spilter(datasetThird, target)
            if dictionary[thirdDecision[0]] == thirdDecision[1]:
                return buy
            else:
                return unbuy
dictionary = {'gender': 'F', 'income': '-10', 'family_number': 1}
print(predicate(dataset,'bought')) 




# gender, income, family_number => <M, -10, 1> output => 1 or 0
