import numpy as np
ls1 = ['hello',True,42,3.14,[1,2,3]]
print(ls1)

s1='anaconda.exe'
filename = s1[:s1.index('.')]
print("",filename)
extension = s1[s1.index('.')+1:]
print("",extension)

import re
s2='ab12cd23ef34ghi45jk'
number=[int(x) for x in s2 if x.isdigit()]
number2=[int(x) for x in re.findall(r'\d+',s2)]
print(number)
print(number2)

import random
score=[]
for i in range(50):
    score.append(random.randint(40,100))
print(score)
for i in range(len(score)):
    if score[i] < 50:
        score[i] = 50
    elif score[i] > 95:
        score[i] = 95
print(score)
first = ['高兴的', '难过的', '伤心的', '搞笑的', '郁闷的', '口渴的', '不好意思的', '生气的', '闹心的', '美丽的', '傻傻的', '踏实的', '不安的', '本分的', '跑得快的', '带不动的']
second = ['安琪拉', '小鲁班', '妲己', '赵云', '扁鹊', '周瑜', '武则天', '露娜', '不知火舞', '诸葛亮', '司马懿', '西施', '芈月', '嫦娥', '程咬金', '吕布']

name = [random.choice(first) + random.choice(second) for i in range(50)]
print(name)

info = dict(zip(name, score))
print(info)
for k, v in info.items():
    if v >= 60:
        print(k)
to_delete=[]
for key in info.keys():
    if '生气的' in key:
        to_delete.append(key)
for key in to_delete:
    info.pop(key)
print(info)
sorted_info = sorted(info.items(),key=lambda x:x[1],reverse=True)
print("前三名:")
for name,score in sorted_info[:3]:
    print(f"{name}:{score}")
print("后三名:")
for name, score in sorted_info[-3:]:
    print(f"{name}:{score}")

