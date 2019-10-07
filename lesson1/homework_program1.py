#这是我的编程实践部分，第一个程序
import jieba
from collections import Counter
corpus = '/Users/jeannewu/Documents/project/nlp_related_exercise/lesson1/train.txt'
FILE = open(corpus.strip(), 'r').read()

#以下是corpus文件的分词情况
def cutOriginal():
    content = FILE.replace('++$++','')
    content1 = content.replace(' ','')
    content2 = content1.replace('\n','')
    return list(jieba.cut(content2))

TOKENS = cutOriginal()
_2_gram_words = [
    TOKENS[i] + TOKENS[i+1] for i in range(len(TOKENS)-1)
]

_2_gram_word_counts = Counter(_2_gram_words)
words_count = Counter(TOKENS)


human_rules = """
human = subject action activity
subject = 我 | 俺 | 我们 
action = 看看 | 找找 | 想找点
activity = 乐子 | 玩的
"""

host_rules = """
host = care numbers enquire business_related retail 
numbers = 我是 number 号 ,
number = single_number | number single_number 
single_number = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 
care = names say_hello | say_hello
names = name ,
name = 先生 | 女士 | 小朋友
say_hello = 你好 | 您好 
enquire = 请问你要 | 您需要
business_related = play concrete_business
play = 耍一耍 | 玩一玩
concrete_business = 喝酒 | 打牌 | 打猎 | 赌博
retail = 吗？"""

import random
hello_rules = '''
say_hello = names hello tail
names = name names | name
name = Jhon | Mike | 老梁
hello = 你好 | 您来了 | 快请进
tail = 呀 | ！
'''



def get_generation_by_gram(grammar_rule, target, stmt_split = '=', or_split='|'):
    rules = dict() # key is the @statement, value is @expression 


    for line in grammar_rule.strip().split('\n'):
        if not line in line: continue
        #skip the empty line
        stmt, expr=line.split(stmt_split)

        # print(stmt, expr.split(or_split))
    
        rules[stmt.strip()] = expr.split(or_split)
        
    # print(rules)

    if target in rules:

        # print(rules[target])
        candidates = rules[target]
        # candidates = [c.strip() for c in candidates]
        candidate = random.choice(candidates)
        # print(candidate)
        words = ''.join(get_generation_by_gram(grammar_rule, target = c) for c in candidate.split())
        # print(words)
        return words
    else: 
        return target




def cut(string):
    return list(jieba.cut(string))

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

# print(two_gram_model('汽车保鲜是否预付？'))

def generate_n(list):
    sortList = dict()
    # print(list)
    a = ''
    for i in list:
        key = get_generation_by_gram(i, list[i])
        value = two_gram_model(key)
        sortList[key] = value

    sorted(sortList, key = lambda x: x[1], reverse=True)
 
    return a 

        
print(generate_n({human_rules: 'human', host_rules: 'host'}))  #入口调用


