import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv('user_data_1503960366.csv')

# 提取特征和目标变量
X = data[['VeryActiveDistance', 'ModeratelyActiveDistance', 'LightActiveDistance']]
y = data['Calories']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建多项式特征
poly = PolynomialFeatures(degree=1)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# 训练多项式回归模型
model = LinearRegression()
model.fit(X_train_poly, y_train)

# 预测
y_pred = model.predict(X_test_poly)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse)

# 绘制预测曲线
plt.scatter(X_test['VeryActiveDistance'], y_test, label='True VeryActiveDistance', alpha=0.3)
plt.scatter(X_test['ModeratelyActiveDistance'], y_test, label='True ModeratelyActiveDistance', alpha=0.3)
plt.scatter(X_test['LightActiveDistance'], y_test, label='True LightActiveDistance', alpha=0.3)

plt.scatter(X_test['VeryActiveDistance'], y_pred, label='Predicted VeryActiveDistance', alpha=0.3)
plt.scatter(X_test['ModeratelyActiveDistance'], y_pred, label='Predicted ModeratelyActiveDistance', alpha=0.3)
plt.scatter(X_test['LightActiveDistance'], y_pred, label='Predicted LightActiveDistance', alpha=0.3)

plt.xlabel("Distances")
plt.ylabel("Calories")
plt.legend()
plt.show()