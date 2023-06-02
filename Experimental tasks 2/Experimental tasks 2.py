import numpy as np
import csv

arr1 = np.random.randint(1, 100, size=(10, 10))
print(arr1)

arr1 = np.random.randint(1, 100, size=(10, 10))

print("形状: ", arr1.shape)
print("维数: ", arr1.ndim)
print("元素个数: ", arr1.size)
print("所占空间大小: ", arr1.nbytes, "bytes")

# 修改左上角5行5列的元素
left_top = np.multiply(np.eye(5), 50) + np.multiply(np.ones((5, 5), dtype=int), 1 - np.eye(5))
arr1[:5, :5] = left_top

# 修改右下角5行5列的元素
right_bottom = np.multiply(np.eye(5, dtype=int), [40, 30, 20, 10, 0]) + np.multiply(np.ones((5, 5), dtype=int), 1 - np.eye(5))
arr1[5:, 5:] = right_bottom

print(arr1)

arr2 = np.random.normal(loc=0, scale=1, size=(10, 10))

print("arr2=",arr2)

for i in range(len(arr2)):
    arr2[i,] += i+1
    
print("arr2_1=",arr2)

# 横向拼接
arr3 = np.concatenate((arr1, arr2), axis=1)
arr3 = arr3.astype(int)

# 纵向拼接
arr4 = np.concatenate((arr1, arr2), axis=0)
arr4 = arr4.astype(int)

print("arr3=",arr3)
print("arr4=",arr4)

# 保存为csv文件
np.savetxt('arr3.csv', arr3, fmt='%d', delimiter=',')

# 保存为txt文件
np.savetxt('arr4.txt', arr4, fmt='%d', delimiter=',')

# 读取csv文件
arr5 = np.loadtxt('arr3.csv', dtype=int, delimiter=',')

# 读取txt文件
arr6 = np.loadtxt('arr4.txt', dtype=int, delimiter=',')

print("arr5=",arr5)
print("arr6=",arr6)

# 对arr5按列进行升序排序
arr5 = np.sort(arr5, axis=0)

print("arr5_sort=",arr5)

# 对arr6按行进行降序排序
arr6 = np.flip(np.sort(arr6, axis=1), axis=0)

print("arr6_sort=",arr6)

# a) 读取数据
filename = "实验2数据.csv"
data = []
with open(filename, "r") as f:
    for line in f.readlines():
        row = line.strip().split("@#")
        data.append(list(map(float, row)))

data = np.array(data)

# b) 计算各科平均分并添加到data中
average_scores = np.mean(data, axis=0)
data = np.vstack([data, average_scores])

# 计算每位同学的平均分并添加到data中
average_student_scores = np.mean(data, axis=1)
data = np.column_stack([data, average_student_scores])

# 四舍五入，保留两位小数
data = np.round(data, 2)

# 保存数据到CSV文件
output_filename = "Achievement.csv"
with open(output_filename, "w", newline="") as f:
    writer = csv.writer(f)
    for row in data:
        writer.writerow(row)

# c) 评选优秀学生
weights = np.array([0.1, 0.2, 0.2, 0.15, 0.35])
passing_score = 60

# 计算权重分
weighted_scores = np.dot(data[:-1, :-1], weights)

# 选择及格的同学
passing_students = np.all(data[:-1, :-1] >= passing_score, axis=1)

# 计算及格同学的权重分和方差
passing_weighted_scores = weighted_scores[passing_students]
var_weighted_scores = np.var(data[:-1, :-1][passing_students], axis=1)

# 根据权重分和方差进行评选
evaluation_scores = passing_weighted_scores - var_weighted_scores
outstanding_students = np.argsort(-evaluation_scores)[:5]

print("优秀学生的序号：", outstanding_students)