import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# 加载数据
data = pd.read_csv('实验4数据.csv')
print(data.isnull().sum())
print(data.info())
print(data.describe())

data["TotalMinutes"] = data["VeryActiveMinutes"] + data["FairlyActiveMinutes"] + data["LightlyActiveMinutes"] + data["SedentaryMinutes"]
data["TotalMinutes"].sample(5)
'''
# 研究每日总步数和消耗的卡路里之间的联系。 
figure = px.scatter(data_frame = data, x="Calories",
                    y="TotalSteps", size="VeryActiveMinutes", 
                    trendline="ols", 
                    title="总步数和消耗的卡路里的关系")
figure.show()

label = ["Very Active Minutes", "Fairly Active Minutes", "Lightly Active Minutes", "Inactive Minutes"]
counts = data[["VeryActiveMinutes", "FairlyActiveMinutes", "LightlyActiveMinutes", "SedentaryMinutes"]].mean()
colors = ["gold","lightgreen", "pink", "blue"]
fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text="总活动时间")
fig.update_traces(hoverinfo="label+percent", textinfo="value", textfont_size=24, marker=dict(colors=colors, line=dict(color="black", width=3)))
fig.show()
'''
data['ActivityDate'] = pd.to_datetime(data['ActivityDate'])
data["Day"] = data["ActivityDate"].dt.day_name()
data["Day"].head()
'''
fig = go.Figure()
fig.add_trace(go.Bar(
                         x=data["Day"],
                         y=data["VeryActiveMinutes"],
                         name="Very Active",
                         marker_color="purple"
                        ))
fig.add_trace(go.Bar(
                         x=data["Day"],
                         y=data["FairlyActiveMinutes"],
                         name="Fairly Active",
                         marker_color="green"
                        ))
fig.add_trace(go.Bar(
                         x=data["Day"],
                         y=data["LightlyActiveMinutes"],
                         name="Lightly Active",
                         marker_color="pink"
                        ))
fig.update_layout(barmode="group", xaxis_tickangle=-45)
fig.show()

day = data["Day"].value_counts()
label = day.index
counts = data["SedentaryMinutes"]
colors = ['gold','lightgreen', "pink", "blue", "skyblue", "cyan", "orange"]
fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Inactive Minutes Daily')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30,
                  marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()

calories = data["Day"].value_counts()
label = calories.index
counts = data["Calories"]
colors = ['gold','lightgreen', "pink", "blue", "skyblue", "cyan", "orange"]
fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Calories Burned Daily')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30, marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()
'''
import seaborn as sns
sns.set(rc={'figure.figsize':(8,6)})
activity_by_week_day = sns.barplot(x="Day", y="TotalSteps", data=data, 
                                   order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], 
                                   capsize=.2)


features = ['TotalSteps', 'TotalDistance', 'TrackerDistance', 'LoggedActivitiesDistance', 'VeryActiveDistance', 'ModeratelyActiveDistance', 'LightActiveDistance', 'SedentaryActiveDistance', 'VeryActiveMinutes', 'FairlyActiveMinutes', 'LightlyActiveMinutes', 'SedentaryMinutes', 'TotalMinutes', 'Day']
target = 'Calories'
# 数据切分
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=0)
# 使用lightgbm训练
from lightgbm import LGBMRegressor
lgbm = LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=0)
# 「星期几」字段编码
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_train['Day'] = le.fit_transform(X_train['Day'])
X_test['Day'] = le.transform(X_test['Day'])
# 拟合模型
lgbm.fit(X_train, y_train)
# 测试集预估
predictions = lgbm.predict(X_test)
# 计算测试集RMSE
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE: %f" % (rmse))

# 使用网格搜索对lightgbm模型进行超参数调优
from sklearn.model_selection import GridSearchCV
parameters = {
                'learning_rate': [0.02, 0.05, 0.08, 0.1],
                'max_depth': [5, 7, 10],
                'feature_fraction': [0.6, 0.8, 0.9],
                'subsample': [0.6, 0.8, 0.9],
                'n_estimators': [100, 200, 500, 1000]}
# 网格搜索
grid_search = GridSearchCV(lgbm, parameters, cv=5, n_jobs=-1, verbose=1)
# 最佳模型
grid_search.fit(X_train, y_train)
best_lgbm = grid_search.best_estimator_
# 输出最佳超参数
print(grid_search.best_params_)
# 测试集预估
predictions = best_lgbm.predict(X_test)
# 计算RMSE
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE: %f" % (rmse))

#绘制特征重要度
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
importance = best_lgbm.feature_importances_
feature_importance = pd.DataFrame({'feature': features, 'importance': importance})
feature_importance = feature_importance.sort_values('importance', ascending=True)
feature_importance.plot.barh(x='feature', y='importance', figsize=(20,10))
'''
# 除去 'Id', 'ActivityDate', 'TotalDistance', 'LoggedActivitiesDistance' 列
data = data.drop(['Id', 'ActivityDate', 'TrackerDistance', 'LoggedActivitiesDistance','SedentaryActiveDistance','SedentaryMinutes','ModeratelyActiveDistance'], axis=1)

# 计算特征与 Calories 之间的皮尔逊相关系数
correlations = data.corr()['Calories'].sort_values(ascending=False)

# 找到与 Calories 列关系最大的至少四个其他列
top_correlation_features = correlations.index[1:8]
print("与 Calories 列关系最大的至少四个其他列:", top_correlation_features)

# 提取特征和目标变量
X = data[top_correlation_features]
y = data['Calories']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定义模型列表
models = [
    ('Linear Regression', LinearRegression()),
    ('KNeighbors Regressor', KNeighborsRegressor()),
    ('Support Vector Regression', SVR()),
    ('Ridge Regression', Ridge()),
    ('Lasso Regression', Lasso()),
    ('MLP Regressor', MLPRegressor(max_iter=1000)),
    ('Decision Tree Regressor', DecisionTreeRegressor()),
    ('Extra Tree Regressor', ExtraTreeRegressor()),
    ('Random Forest Regressor', RandomForestRegressor()),
    ('AdaBoost Regressor', AdaBoostRegressor()),
    ('Gradient Boosting Regressor', GradientBoostingRegressor()),
    ('Bagging Regressor', BaggingRegressor()),
]

# 用不同的模型进行分析
for model_name, model in models:
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{model_name} - Mean Squared Error: {mse}")
'''