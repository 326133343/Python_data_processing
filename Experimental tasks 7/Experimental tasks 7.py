import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer

#读取 csv 文件
data = pd.read_csv('实验7数据.csv', encoding='gbk')

#将 Temp 列中的缺失值替换为 25，删除 Humi 列中有缺失值的行
data['Temp'].fillna(25, inplace=True)
data.dropna(subset=['Humi'], inplace=True)

#使用前后两个相邻值的平均值填充 Fog 列中的缺失值
data['Fog'] = data['Fog'].fillna((data['Fog'].shift() + data['Fog'].shift(-1)) / 2)

#删除数据中的重复行
data.drop_duplicates(inplace=True)

#使用 corr() 函数分析数值列的相关性
numerical_data = data[['Temp', 'Humi', 'Fog']]
correlation = numerical_data.corr()
print(correlation)

#对 Temp 和 Humi 列进行标准差标准化，对 Fog 列进行离差标准化
scaler = StandardScaler()
data[['Temp', 'Humi']] = scaler.fit_transform(data[['Temp', 'Humi']])

min_max_scaler = MinMaxScaler()
data['Fog'] = min_max_scaler.fit_transform(data['Fog'].values.reshape(-1, 1))

#替换 Target 列中的值
data['Target'] = data['Target'].map({'正常': 0, '一般': 1, '严重': 2})

#创建一个基于 SVC 的分类模型
X = data[['Temp', 'Humi', 'Fog']]
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建imputer对象，用于填充缺失值
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# 对训练集进行填充
X_train = imputer.fit_transform(X_train)

# 对测试集进行填充
X_test = imputer.transform(X_test)

model = SVC()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))