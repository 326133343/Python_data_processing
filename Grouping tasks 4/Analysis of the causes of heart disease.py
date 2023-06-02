import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
import joblib
from collections import Counter

df = pd.read_csv('heart_2020_cleaned.csv')

#数据集清洗
df = df[(df['SleepTime'] >= 4) & (df['SleepTime'] <= 14)]

df.to_csv('heart_2020_cleaned.csv', index=False)

#BMI箱线图
sns.boxplot(df['BMI'])
plt.show()

# BMI分类饼图
bmi_categories = ['Underweight', 'Normal weight', 'Overweight', 'Obese']
df['BMI_Category'] = pd.cut(df['BMI'], bins=[0,18.5,24.9,29.9,100], labels=bmi_categories, right=False)
bmi_counts = df['BMI_Category'].value_counts()
plt.pie(bmi_counts, labels = bmi_counts.index)
plt.show()

# 对SleepTime列进行处理，将数值转化为词
sleep_hours = df['SleepTime'].apply(lambda x: '{}H'.format(int(x))).tolist()

# 将所有的词组合成一个字符串
sleep_words = ' '.join(sleep_hours)

# 生成词云
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                min_font_size = 10).generate(sleep_words)

# 绘制词云
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()
#年龄段人数柱状图
# 计算每个年龄段的频率
age_counts = df['AgeCategory'].value_counts().sort_index(ascending=True)

# 绘制柱状图
age_counts.plot(kind='bar')
plt.xlabel('Age group')
plt.ylabel('Number of people')
plt.title('Number of people in each age group')
plt.show()

#吸烟与心脏病间关系堆积图
smoking_vs_disease = df.groupby('Smoking')['HeartDisease'].value_counts().unstack()
smoking_vs_disease.plot(kind='bar', stacked=True)
plt.show()

# 清理缺失值
df_cleaned = df.dropna(subset=['SleepTime', 'PhysicalActivity', 'AlcoholDrinking', 'HeartDisease'])

# 计算每个睡眠时长与心脏病间关系堆积图
heart_disease_counts = df[df['HeartDisease'] == 'Yes'].groupby('SleepTime').size()
no_heart_disease_counts = df[df['HeartDisease'] == 'No'].groupby('SleepTime').size()

# 创建堆积柱状图
plt.figure(figsize=(10,6))
p1 = plt.bar(heart_disease_counts.index, heart_disease_counts.values)
p2 = plt.bar(no_heart_disease_counts.index, no_heart_disease_counts.values, bottom=heart_disease_counts.values)

plt.ylabel('Number of People')
plt.xlabel('SleepTime')
plt.title('Number of People by SleepTime and HeartDisease')
plt.xticks(range(int(df['SleepTime'].min()), int(df['SleepTime'].max())+1))
plt.legend((p1[0], p2[0]), ('Heart Disease', 'No Heart Disease'))

plt.show()

# 绘制锻炼情况与心脏病的关联性图
plt.figure(figsize=(8, 6))
sns.countplot(x='PhysicalActivity', hue='HeartDisease', data=df_cleaned)
plt.title('Association between exercise status and heart disease')
plt.show()

# 绘制饮酒与心脏病的关联性图
plt.figure(figsize=(8, 6))
sns.countplot(x='AlcoholDrinking', hue='HeartDisease', data=df_cleaned)
plt.title('The association between alcohol consumption and heart disease')
plt.show()

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

print(df.head())

# 进行相关性分析
correlation_matrix = df.corr()
print(correlation_matrix["HeartDisease"].sort_values(ascending=False))

df.to_csv('heart_2020_cleaned_After pre-treatment.csv', index=False)