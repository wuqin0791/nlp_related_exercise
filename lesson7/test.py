'''
@Description: This is a python file
@Author: JeanneWu
@Date: 2019-11-30 16:06:51
'''
import torch
import torch.nn as nn
from torch.autograd import Variable

#构建网络模型---输入矩阵特征数input_size、输出矩阵特征数hidden_size、层数num_layers
inputs = torch.randn(5,3,10)  # ->(seq_len,batch_size,input_size)
rnn = nn.LSTM(10,20,2)   # ->   (input_size,hidden_size,num_layers)
h0 = torch.randn(2,3,20)  # ->(num_layers* 1,batch_size,hidden_size)
c0 = torch.randn(2,3,20)  # ->(num_layers*1,batch_size,hidden_size) 
num_directions=1 #因为是单向LSTM
'''
Outputs: output, (h_n, c_n)
'''
output,(hn,cn) = rnn(inputs,(h0,c0))

print(output)