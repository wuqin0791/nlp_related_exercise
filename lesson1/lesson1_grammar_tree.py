#这是课堂代码复现

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
        print(words)
        return words
    else: 
        return target


get_generation_by_gram(hello_rules, target='say_hello')