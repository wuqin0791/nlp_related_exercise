'''
@Description: 这是编程题第二题
@Author: JeanneWu
@Date: 2019-11-03 08:45:05
将上一节课(第二节课)的线性回归问题中的Loss函数改成"绝对值"，并且改变其偏导的求值方式，观察其结果的变化。(19 point)
是否将Loss改成了“绝对值”(3')
是否完成了偏导的重新定义(5')
新的模型Loss是否能够收敛 (11’)
'''
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
# modify loss function to absolute
def loss(y,y_hat):
    return sum(abs(y_i - y_hat_i)for y_i, y_hat_i in zip(list(y),list(y_hat)))

# define partial derivative 
# 对k求偏导得到 sum(abs(yi-xi))
# 对b求偏导得到 sum(abs(yi-1))
def partial_derivative_k(x, y, y_hat):
    n = len(y)
    gradient = 0
    for x_i, y_i in zip(list(x),list(y)):
        if y_i > x_i:
            gradient += (y_i-x_i)
        else:
            gradient += (x_i-y_i)
    return gradient

def partial_derivative_b(y, y_hat):
    n = len(y)
    gradient = 0
    for y_i, y_hat_i in zip(list(y),list(y_hat)):
        if y_i > 1:
            gradient += (y_i-1)
        else: 
            gradient += (1-y_i)
    return gradient

k = random.random() * 200 - 100  # -100 100
b = random.random() * 200 - 100  # -100 100

learning_rate = 1e-4

iteration_num = 500 
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

#loss function最终没有收敛
    