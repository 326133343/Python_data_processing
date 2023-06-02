import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

#数据读取
data = pd.read_csv('实验6数据.txt', header=None, sep=',')

#统计各班各分数段人数
bins = [0, 60, 70, 80, 90, 100]
class_data = [np.histogram(data[i], bins=bins)[0] for i in range(4)] # 4个班级的数据
y1, y2, y3, y4 = class_data

#1班各分数段人数分布柱状图
plt.figure(figsize=(8, 6))
plt.bar(range(len(y1)), y1)
plt.title('Histogram of the distribution of the number of students in each score band of class 1')
plt.show()

#1班的各分数段人数占比分布饼图
plt.figure(figsize=(8, 6))
plt.pie(y1, autopct='%1.1f%%')
plt.title('Pie chart of the distribution of the number of students in each score band in class 1')
plt.show()

#1到4班的分数分布柱状图
plt.figure(figsize=(10, 8))
labels = ['<60', '60-70', '70-80', '80-90', '90-100']
x = np.arange(len(labels)) 
width = 0.2
rects1 = plt.bar(x - 3/2*width, y1, width, label='class1')
rects2 = plt.bar(x - width/2, y2, width, label='class2')
rects3 = plt.bar(x + width/2, y3, width, label='class3')
rects4 = plt.bar(x + 3/2*width, y4, width, label='class4')

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2., 1.02*height,
                '%d' % int(height), ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

plt.ylabel('Number of people')
plt.title('Histogram of score distribution of classes 1 to 4')
plt.xticks(x, labels)
plt.legend()
plt.show()

#1到4班的分数分布箱线图
plt.figure(figsize=(10, 8))
plt.boxplot([data[i].values for i in range(4)], labels=['class1', 'class2', 'class3', 'class4'])
plt.title('Box plot of score distribution for classes 1 to 4')
plt.show()

#1班各分数段人数柱线图
plt.figure(figsize=(8, 6))
plt.bar(range(len(y1)), y1, color='skyblue')
plt.plot(range(len(y1)), y1, marker='o', color='r')
for i in range(len(y1)):
    plt.text(i, y1[i], y1[i], ha='center', va='bottom')
plt.title('Histogram of the number of students in each score band of class 1')
plt.show()

#子图
fig, ax = plt.subplots(2, 3, figsize=(18, 12))

#各班分数的折线图
for i, y in enumerate(class_data, start=1):
    ax[0, 0].plot(range(len(y)), y, marker='o', label=f'class{i}')
ax[0, 0].set_title('Line graph of the scores of each class')
ax[0, 0].legend()

# 统计各班各个分组的人数
x = np.arange(len(labels)) 
width = 0.2
rects1 = ax[1, 0].bar(x - 3/2*width, y1, width, label='class1')
rects2 = ax[1, 0].bar(x - width/2, y2, width, label='class2')
rects3 = ax[1, 0].bar(x + width/2, y3, width, label='class3')
rects4 = ax[1, 0].bar(x + 3/2*width, y4, width, label='class4')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

ax[1, 0].set_ylabel('Number of people')
ax[1, 0].set_title('Histogram of the distribution of the number of students in each class by score')
ax[1, 0].set_xticks(x)
ax[1, 0].set_xticklabels(labels)
ax[1, 0].legend()

# 1到4班的饼图
for i, y in enumerate(class_data, start=1):
    ax[(i-1)//2, (i-1)%2+1].pie(y, autopct='%1.1f%%')
    ax[(i-1)//2, (i-1)%2+1].set_title(f'class {i}')

plt.tight_layout()
plt.show()

#二维图像三维化

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'g', 'b', 'y']
labels = ['<60', '60-70', '70-80', '80-90', '90-100']

for c, k in zip(colors, range(4)):
    xs = np.arange(5)
    ys = class_data[k]
    
    ax.bar(xs, ys, zdir='y', zs=k, color=c, alpha=0.8)

ax.set_xlabel('Score Segment Grouping')
ax.set_ylabel('class')
ax.set_zlabel('Number of people')

# 设置y轴的刻度和标签
ax.set_yticks(np.arange(4))
ax.set_yticklabels(['class1', 'class2', 'class3', 'class4'])

# 设置x轴的刻度和标签
ax.set_xticks(np.arange(5))
ax.set_xticklabels(labels)

ax.view_init(elev=20., azim=-35)

plt.show()