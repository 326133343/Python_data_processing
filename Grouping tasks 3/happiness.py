import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
data = pd.read_csv('幸福感数据集部分特征数据.csv')

# 删除所有列中含有小于0的数值的行
data = data[(data >= 0).all(1)]

# 计算当前年龄
data['age'] = 2023 - data['birth']

# 删除65岁以上的数据
data = data[data['age'] <= 65]

# 将所有income列中大于1000但是小于10000的值都乘以12
data.loc[(data['income'] > 1000) & (data['income'] < 10000), 'income'] *= 12

# 将edu中按数值分组，分别计算自己这一组中所有income列数值大于10000的平均值，然后替换自己这组中income列小于10000的值
for edu_class in data['edu'].unique():
    mean_income = data[(data['edu'] == edu_class) & (data['income'] > 10000)]['income'].mean()
    data.loc[(data['edu'] == edu_class) & (data['income'] < 10000), 'income'] = mean_income

# 保存处理后的CSV文件
data.to_csv('幸福感数据集处理后.csv', index=False)

# 读取处理过的CSV文件
data = pd.read_csv('幸福感数据集处理后.csv')

# 计算每个年龄段和教育等级的人数
age_counts = data['age'].value_counts().sort_index()
edu_counts = data['edu'].value_counts().sort_index()

# 设置绘图风格
sns.set(style="whitegrid")

# 绘制年龄分布的直方图
sns.barplot(x=age_counts.index, y=age_counts.values, color="blue")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of People")
plt.show()

# 绘制收入与年龄的箱线图
sns.boxplot(data=data, x="age", y="income")
plt.title("Income Distribution by Age")
plt.xlabel("Age")
plt.ylabel("Income")
plt.show()

# 绘制教育程度分布的直方图
sns.barplot(x=edu_counts.index, y=edu_counts.values, color="green")
plt.title("Education Level Distribution")
plt.xlabel("Education Level")
plt.ylabel("Number of People")
plt.show()

# 绘制身高与体重的散点图
sns.scatterplot(data=data, x="height_cm", y="weight_jin")
plt.title("Height vs. Weight")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (jin)")
plt.show()

# 绘制收入均值与中位数的折线图
age_groups = data.groupby('age')['income']
mean_income_by_age = age_groups.mean()
median_income_by_age = age_groups.median()

plt.plot(mean_income_by_age.index, mean_income_by_age.values, label='Mean Income', marker='o')
plt.plot(median_income_by_age.index, median_income_by_age.values, label='Median Income', marker='o')
plt.title("Mean and Median Income by Age")
plt.xlabel("Age")
plt.ylabel("Income")
plt.legend()
plt.show()

# 创建一个3x2的子图布局
fig, axes = plt.subplots(3, 2, figsize=(20, 18))

# 绘制年龄分布的直方图
sns.barplot(x=age_counts.index, y=age_counts.values, color="blue", ax=axes[0, 0])
axes[0, 0].set_title("Age Distribution")
axes[0, 0].set_xlabel("Age")
axes[0, 0].set_ylabel("Number of People")

# 绘制收入与年龄的箱线图
sns.boxplot(data=data, x="age", y="income", ax=axes[0, 1])
axes[0, 1].set_title("Income Distribution by Age")
axes[0, 1].set_xlabel("Age")
axes[0, 1].set_ylabel("Income")

# 绘制教育程度分布的直方图
sns.barplot(x=edu_counts.index, y=edu_counts.values, color="green", ax=axes[1, 0])
axes[1, 0].set_title("Education Level Distribution")
axes[1, 0].set_xlabel("Education Level")
axes[1, 0].set_ylabel("Number of People")

# 绘制身高与体重的散点图
sns.scatterplot(data=data, x="height_cm", y="weight_jin", ax=axes[1, 1])
axes[1, 1].set_title("Height vs. Weight")
axes[1, 1].set_xlabel("Height (cm)")
axes[1, 1].set_ylabel("Weight (jin)")

# 绘制收入均值与中位数的折线图
axes[2, 0].plot(mean_income_by_age.index, mean_income_by_age.values, label='Mean Income', marker='o')
axes[2, 0].plot(median_income_by_age.index, median_income_by_age.values, label='Median Income', marker='o')
axes[2, 0].set_title("Mean and Median Income by Age")
axes[2, 0].set_xlabel("Age")
axes[2, 0].set_ylabel("Income")
axes[2, 0].legend()

# 删除多余的子图
fig.delaxes(axes[2, 1])

# 调整子图间距
plt.subplots_adjust(wspace=0.3, hspace=0.3)

# 保存图形为jpg格式，dpi为600
plt.savefig('combined_plots_with_income_line.jpg', dpi=600)

# 显示图形
plt.show()