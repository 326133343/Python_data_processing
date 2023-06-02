from datetime import datetime, timedelta

# 打开文件
with open('dst2017', 'r') as f:
    # 读取所有行
    lines = f.readlines()

# 创建一个空的二维数组
data = []

# 将起始日期转换为datetime对象
date_str = '20170101'
date = datetime.strptime(date_str, '%Y%m%d')

# 遍历每一行
for line in lines:
    # 将行分割为多个单元格，并转换为正确的数据类型
    row = []
    for x in line.split():
        if x.isdigit():
            row.append(int(x))
        elif '.' in x:
            row.append(int(x))
        else:
            row.append(x)

    # 切出第2到第27列的数据
    row = row[2:27]

    # 在行的开头添加日期，并将日期递增一天
    row.insert(0, date.strftime('%Y%m%d'))
    date += timedelta(days=1)

    # 将行添加到二维数组中
    data.append(row)

# 打印二维数组
print(data)