import sys
sys.path.append(r"C:\Users\32613\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv('实验4数据.csv')

# 提取特征和目标变量
X = data[['VeryActiveDistance', 'ModeratelyActiveDistance', 'LightActiveDistance']]
y = data['Calories']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 加载模型
loaded_model = load_model("my_model.h5")

# 继续训练模型
loaded_model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2)

# 保存继续训练后的模型
loaded_model.save("my_model.h5")

# 预测
y_pred = loaded_model.predict(X_test_scaled)

# 评估模型
mse = tf.keras.losses.MeanSquaredError()
loss = mse(y_test, y_pred)
print("Mean Squared Error: ", loss.numpy())
'''
# 一个新的用户数据点
new_user_data = np.array([[VeryActiveDistance, ModeratelyActiveDistance, LightActiveDistance]])

# 使用训练好的模型进行预测
new_user_data_scaled = scaler.transform(new_user_data)
predicted_calories = model.predict(new_user_data_scaled)

# 输出预测结果
print("预测消耗的卡路里: ", predicted_calories[0][0])
'''
# 可视化预测结果和真实值之间的关系
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Calories")
plt.ylabel("Predicted Calories")
plt.title("Actual vs Predicted Calories")
plt.show()