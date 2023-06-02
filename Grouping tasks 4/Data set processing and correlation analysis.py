import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
import joblib
from collections import Counter

# 读取数据
df = pd.read_csv("heart_2020_cleaned.csv")

# 将二元分类变量编码为0和1
binary_variables = ["HeartDisease", "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking", "Diabetic", "PhysicalActivity", "Asthma", "KidneyDisease", "SkinCancer"]
for var in binary_variables:
    df[var] = df[var].map({"No": 0, "Yes": 1})

# 将性别编码为0和1
df["Sex"] = df["Sex"].map({"Female": 0, "Male": 1})

# 对多元分类变量进行独热编码
df = pd.get_dummies(df, columns=["Race", "GenHealth"])

# 处理年龄段变量
age_mapping = {
    '18-24': 21,
    '25-29': 27,
    '30-34': 32,
    '35-39': 37,
    '40-44': 42,
    '45-49': 47,
    '50-54': 52,
    '55-59': 57,
    '60-64': 62,
    '65-69': 67,
    '70-74': 72,
    '75-79': 77,
    '80+': 85
}
df['AgeCategory'] = df['AgeCategory'].map(age_mapping)

# 对连续变量进行标准化
scaler = StandardScaler()
continuous_variables = ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"]
df[continuous_variables] = scaler.fit_transform(df[continuous_variables])

# 获取患病和未患病的样本
positive = df[df['HeartDisease'] == 1]
negative = df[df['HeartDisease'] == 0]

# 对未患病的样本进行随机欠采样
negative_under = negative.sample(len(positive), random_state=42)

# 合并欠采样的未患病样本和患病样本
df = pd.concat([positive, negative_under], axis=0)

# 设定随机种子，确保结果的可复现性
random_state = 42

# 分离特征和标签
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# 创建imputer对象，用于填充缺失值
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# 对训练集进行拟合和转换
X_train_imputed = imputer.fit_transform(X_train)

# 对测试集进行转换
X_test_imputed = imputer.transform(X_test)

# 随机森林模型
rf = RandomForestClassifier(random_state=random_state)
rf.fit(X_train_imputed, y_train)

# 模型性能评估
rf_y_pred = rf.predict(X_test_imputed)
print("Random Forest model performance:")
print(classification_report(y_test, rf_y_pred))

# 梯度提升模型
gb = GradientBoostingClassifier(random_state=random_state)
gb.fit(X_train_imputed, y_train)

# 模型性能评估
gb_y_pred = gb.predict(X_test_imputed)
print("Gradient Boosting model performance:")
print(classification_report(y_test, gb_y_pred))