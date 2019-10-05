import jieba

corpus = '/Users/jeannewu/Documents/project/nlp_related_exercise/lesson1/article_9k.txt'
FILE = open(corpus).read() #读文件
def generate_by_pro(text_corpus, length):
    return ''.join(random.sample(text_corpus, length))
# print(len(FILE))

print(FILE[:500])

max_length = 1000000
sub_file = FILE[:max_length]

def cut(string):
    return list(jieba.cut(string))

TOKENS = cut(sub_file) #? 什么情况下赋值函数要大写？
print(len(TOKENS))


