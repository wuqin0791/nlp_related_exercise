'''
@Description: This is a python file
@Author: JeanneWu
@Date: 2019-10-28 11:21:45
'''
import collections
s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]

d = collections.defaultdict(list)
print(d)
for k, v in s:
    d[k].append(v)

list(d.items())
#[('yellow', [1, 3]), ('blue', [2, 4]), ('red', [1])]

#第二个例子
s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]

# defaultdict
d = collections.defaultdict(list)
for k, v in s:
    d[k].append(v)

# Use dict and setdefault    
g = {}
for k, v in s:
    g.setdefault(k, []).append(v)
    


# Use dict
e = {}
for k, v in s:
    e[k] = v


##list(d.items())
##list(g.items())
##list(e.items())