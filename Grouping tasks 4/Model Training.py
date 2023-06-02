import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import joblib
from sklearn.cluster import KMeans

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

# 设定随机种子，确保结果的可复现性
random_state = 42

# 分离特征和标签
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# 为特征数据添加列名
X.columns = df.columns.drop("HeartDisease")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# 创建imputer对象，用于填充缺失值
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# 对训练集进行填充
X_train = imputer.fit_transform(X_train)
# 对测试集进行填充
X_test = imputer.transform(X_test)

# 对训练集进行聚类
kmeans = KMeans(n_clusters=2, random_state=random_state)
cluster_labels = kmeans.fit_predict(X_train)

# 根据聚类结果删除一部分未患病的样本
mask = (cluster_labels == 0) | (y_train == 1)
X_train_balanced = X_train[mask]
y_train_balanced = y_train[mask]

# 创建和训练模型
models = {
    "Random Forest": RandomForestClassifier(random_state=random_state),
    "Gradient Boosting": GradientBoostingClassifier(random_state=random_state)
}

for name, model in models.items():
    model.fit(X_train_balanced, y_train_balanced)
    predictions = model.predict(X_test)
    print(f"{name} model performance:")
    print(classification_report(y_test, predictions))

# 设定模型参数范围
rf_params = {
    'n_estimators': [100],
    'max_depth': [None, 2]
}
gb_params = {
    'n_estimators': [100],
    'learning_rate': [0.1],
    'max_depth': [3]
}

# 创建模型
rf_model = RandomForestClassifier(random_state=random_state)
gb_model = GradientBoostingClassifier(random_state=random_state)

# 创建GridSearchCV对象，进行模型参数搜索
rf_grid = GridSearchCV(rf_model, rf_params, cv=3)
gb_grid = GridSearchCV(gb_model, gb_params, cv=3)

# 对训练集进行拟合
rf_grid.fit(X_train_balanced, y_train_balanced)
gb_grid.fit(X_train_balanced, y_train_balanced)

# 输出最优参数
print("Random Forest best parameters:", rf_grid.best_params_)
print("Gradient Boosting best parameters:", gb_grid.best_params_)

# 对测试集进行预测
rf_predictions = rf_grid.predict(X_test)
gb_predictions = gb_grid.predict(X_test)

# 输出模型性能
print("Random Forest model performance:")
print(classification_report(y_test, rf_predictions))
print("Gradient Boosting model performance:")
print(classification_report(y_test, gb_predictions))

# 保存训练好的模型
joblib.dump(rf_grid.best_estimator_, 'rf_model.pkl')
joblib.dump(gb_grid.best_estimator_, 'gb_model.pkl')