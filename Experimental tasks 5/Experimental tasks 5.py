import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

data = pd.read_excel('实验5数据.xlsx')

#删除有缺省值的行
data = data.dropna()

#删除获赞数、播放数、平均播放数、平均获赞数、总粉丝人数、充电人数和作品数中小于等于0的行
columns_to_check = ['获赞数', '播放数', '平均播放数', '平均获赞数', '总粉丝人数', '充电人数', '作品数']
for column in columns_to_check:
    data = data[data[column] > 0]

#将性别列中的三种不同取值分别用数值替换，其中男替换成0、女替换成1、保密替换成2
data['性别'] = data['性别'].map({'男': 0, '女': 1, '保密': 2})

# 选择数据类型为数值型的列
numerical_data = data.select_dtypes(include=[np.number])

# 计算相关性矩阵
corr_matrix = numerical_data.corr()

# 打印相关性大于0.7的列
for i in range(corr_matrix.shape[0]):
    for j in range(i+1, corr_matrix.shape[1]):
        if corr_matrix.iloc[i, j] > 0.7:
            print(corr_matrix.columns[i], corr_matrix.columns[j])

#使用箱线图的方法，将粉丝数列的异常值打印出来
sns.boxplot(x=data['总粉丝人数'])
plt.show()
z_scores = stats.zscore(data['总粉丝人数'])
outliers = data['总粉丝人数'][np.abs(z_scores) > 3]
print(outliers)

#使用离差标准化对获赞数列进行数据标准化
data['获赞数'] = (data['获赞数'] - data['获赞数'].min()) / (data['获赞数'].max() - data['获赞数'].min())

#使用标准差标准化对播放数列进行数据标准化
data['播放数'] = (data['播放数'] - data['播放数'].mean()) / data['播放数'].std()

#定义：获赞效率为平均每1000次播放的获赞数，为data添加获赞效率列；
data['获赞效率'] = data['获赞数'] / data['播放数'] * 1000

#定义：充电效率为平均每1000次播放的充电数，为data添加充电效率列；
data['充电效率'] = data['充电人数'] / data['播放数'] * 1000

# 打印男性占比最大和女性占比最大的创作领域
# 首先根据创作领域和性别进行分组
grouped = data.groupby(['创作领域', '性别'])

# 计算每个分组内的 UP 主数量
gender_counts = grouped.size().unstack()

# 计算男女比例
gender_ratio = gender_counts.div(gender_counts.sum(axis=1), axis=0)

# 打印男性占比最大和女性占比最大的创作领域
print('男性占比最大的创作领域：', gender_ratio[0].idxmax())
print('女性占比最大的创作领域：', gender_ratio[1].idxmax())

data = data.dropna(subset=['up主'])
data['昵称长度'] = data['up主'].apply(len)
print(data['昵称长度'])

data['获赞效率'] = data['获赞数'] / data['播放数'] * 1000
data['充电效率'] = data['充电人数'] / data['播放数'] * 1000
data['作品效率'] = data['充电人数'] * data['获赞数'] / data['作品数']
print('获赞效率为',data['获赞效率'])
print('充电效率为',data['充电效率'])
print('作品效率为',data['作品效率'])

from sklearn.preprocessing import MinMaxScaler

# 创建一个MinMaxScaler对象
scaler = MinMaxScaler()

# 将需要归一化的列名放入一个列表中
columns_to_normalize = ['获赞效率', '充电效率', '作品效率']

# 使用MinMaxScaler进行归一化处理
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

# 定义计算分数的函数
def calculate_score(row):
    score = 0.3 * row['获赞效率'] + 0.3 * row['作品效率'] + 0.4 * row['充电效率']
    return score

# 计算每行的分数并将其添加为新的列
data['分数'] = data.apply(calculate_score, axis=1)

# 根据类型进行分组，并找到每个类型下分数最高的UP主
grouped = data.groupby('类型')
top_ups = grouped['分数'].idxmax()
top_ups_data = data.loc[top_ups]

print(top_ups_data)