#这是我的program2,请直接看program1，我已经合并了。
import jieba
from collections import Counter

corpus = '/Users/jeannewu/Documents/project/nlp_related_exercise/lesson1/train.txt'
FILE = open(corpus.strip(), 'r').read()

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

print(two_gram_model('汽车保鲜是否预付？'))






