'''
@Description: This is a python file
@Author: JeanneWu
@Date: 2019-10-05 17:09:10
'''
#这是课堂代码复现

import jieba
import random
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

corpus = '/Users/jeannewu/Documents/project/nlp_related_exercise/lesson1/article_9k.txt'
FILE = open(corpus).read() #读文件
def generate_by_pro(text_corpus, length):
    return ''.join(random.sample(text_corpus, length))
# print(len(FILE))

# print(FILE[:500])

max_length = 1000000
sub_file = FILE[:max_length]

def cut(string):
    return list(jieba.cut(string))

TOKENS = cut(sub_file) #? 什么情况下赋值函数要大写？
# print(len(TOKENS))

# %matplotlib inline

words_count = Counter(TOKENS) #计算文字的个数，用collections里的counter来处理
(words_count.most_common(20))

words_with_fre = [f for w, f in words_count.most_common()] #计算出每个分词的频率
# print(words_with_fre,'****')
# print(words_with_fre[:10])

plt.plot(np.log(np.log(words_with_fre))) #为什么py里看不到图片

list(jieba.cut('一加手机5要做市面最轻薄'))

_2_gram_words = [
    TOKENS[i] + TOKENS[i+1] for i in range(len(TOKENS)-1)
]
# print(_2_gram_words, '_2_gram_words == ')
_2_gram_words[:10]
_2_gram_word_counts = Counter(_2_gram_words)

(words_count.most_common()[-1][-1]) ##为什么要取最后一项的最后一个值？

def get_gram_count(word, wc):
    if word in wc: return wc[word]
    else:
        return wc.most_common()[-1][-1]

def two_gram_model(sentence):
    tokens = cut(sentence)

    probability = 1

    for i in range(len(tokens)-1):
        word = tokens[i]
        next_word = tokens[i+1]

        _two_gram_c = get_gram_count(word+next_word, _2_gram_word_counts) #计算两个都出现的个数
        _one_gram_c = get_gram_count(next_word, words_count)#“今天”这个词总数
        pro = _two_gram_c/_one_gram_c

        probability *= pro

    return probability
a = two_gram_model('此外自本周6月12日起除小米手机6等15款机型')

print(a)




