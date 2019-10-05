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



    for line in hello_rules.strip().split('\n'):
        if not line in line: continue
        #skip the empty line
        stmt, expr=line.split(stmt_split)

        # print(stmt, expr.split(or_split))
    
        rules[stmt.strip()] = expr.split(or_split)

    if target in grammar_rule:
        candidates = grammar_rule[target]
        # candidates = [c.strip() for c in candidates]
        candidate = random.choice(candidates)
        # print(candidate)
        return ''.join(generate(grammar_rule, target = c) for c in candidate.split())
    else: 
        return target

print(get_generation_by_gram(rules, target='say_hello'))


def name():
    return random.choice('Jhon | Mike | 老梁'.split('|'))

def hello():
    return random.choice('你好 | 您来了 | 快请进'.split('|'))

def say_hello():
    return name() + '' + hello()

# print(say_hello())