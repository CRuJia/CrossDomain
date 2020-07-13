with open("info.txt") as f:
    a = f.read()
a = a.split('\n')
from collections import defaultdict
dic = defaultdict(str)
for i in a:
    c = i.split(' ')
    k,v = c[0], c[1]

    dic[k] = v


with open("mini.txt") as f:
    a = f.read()

a = a.split(", ")
print(len(a))
res = []
for i in a:
    res.append(dic[i])
print(res)

