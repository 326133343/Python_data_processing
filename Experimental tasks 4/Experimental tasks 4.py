import pandas as pd
import numpy as np


# 使用pandas读取CSV文件
data = pd.read_csv('实验4数据.csv')

# 查看data的index属性
print("Index:")
print(data.index)

# 查看data的columns属性
print("\nColumns:")
print(data.columns)

# 查看data的values属性
print("\nValues:")
print(data.values)

# 查看data的dtypes属性
print("\nData Types:")
print(data.dtypes)

# 显示数据的前10行，以确保读取成功
print(data.head(10))

# 显示数据的后15行，以确保读取成功
print(data.tail(15))

# 使用loc选取指定行和列的数据
print("使用loc选取数据：")
print(data.loc[100:150, data.columns[[1, 3, 5, 7]]])

# 使用iloc选取指定行和列的数据
print("\n使用iloc选取数据：")
print(data.iloc[100:151, [1, 3, 5, 7]])

# 获取第一行的Id号
first_row_id = data.iloc[0]['Id']

# 选择与第一行具有相同Id号的所有行
same_id_rows = data[data['Id'] == first_row_id]

# 打印具有相同Id号的所有行
print("具有相同Id号的所有行：")
print(same_id_rows)

# 获取第一位用户的ID
first_user_id = data.iloc[0]['Id']

# 选取第一位用户的数据
first_user_data = data[data['Id'] == first_user_id]

# 选择第一位用户步数大于10000步的数据
steps_greater_than_10000 = first_user_data[first_user_data['TotalSteps'] > 10000]

# 打印第一位用户步数大于10000步的数据
print("第一位用户步数大于10000步的数据：")
print(steps_greater_than_10000)

# 统计不同Id的数量
unique_user_count = data['Id'].nunique()

# 打印用户数量
print("一共有{}位用户。".format(unique_user_count))

# 删除LoggedActivitiesDistance列
data = data.drop(columns=['LoggedActivitiesDistance'])

# 计算VeryActiveSpeed，将距离（假设单位为公里）转换为米，将时间（假设单位为分钟）转换为秒
data['VeryActiveSpeed'] = (data['VeryActiveDistance'] * 1000) / (data['VeryActiveMinutes'] * 60)

# 填充可能产生的NaN值（当VeryActiveMinutes为0时）
data['VeryActiveSpeed'] = data['VeryActiveSpeed'].fillna(0)

# 显示数据的前5行，以确保新列已被添加
print(data.head())

# 假设“TotalDistance”是总路程列的名称，请根据你的数据集进行调整
# 计算每位用户的总路程和总步数
user_total_distance = data.groupby('Id')['TotalDistance'].sum()
user_total_steps = data.groupby('Id')['TotalSteps'].sum()

# 计算每位用户的平均步长（单位：米/步）
average_step_length = user_total_distance * 1000 / user_total_steps

# 打印每位用户的平均步长
print("每位用户的平均步长（单位：米/步）：")
print(average_step_length)

# 将Date列转换为datetime类型
data['ActivityDate'] = pd.to_datetime(data['ActivityDate'])

# 提取月份和日期，并将其组合成新的值
data['ActivityDate'] = data['ActivityDate'].apply(lambda x: x.month * 100 + x.day)

# 显示数据的前5行，以确保转换成功
print(data.head())

# 找到所有唯一的用户ID
unique_user_ids = data['Id'].unique()

# 遍历所有用户ID
for user_id in unique_user_ids:
    # 从数据集中提取属于该ID的数据
    user_data = data[data['Id'] == user_id]
    
    # 将提取出的数据存储到一个单独的CSV文件中
    user_data.to_csv(f'user_data_{user_id}.csv', index=False)

    # 读取CSV文件
    user_data = pd.read_csv(f'user_data_{user_id}.csv')
    
    # 删除SedentaryActiveDistance列
    user_data = user_data.drop('SedentaryActiveDistance', axis=1)
    
    # 删除Calories列值为0的行
    user_data = user_data[user_data['Calories'] != 0]
    
    # 删除SedentaryActiveDistance列
    user_data = user_data.drop('VeryActiveSpeed', axis=1)

    # 删除SedentaryActiveDistance列
    user_data = user_data.drop('TrackerDistance', axis=1)
    
    # 将修改后的数据写回CSV文件
    user_data.to_csv(f'user_data_{user_id}.csv', index=False)