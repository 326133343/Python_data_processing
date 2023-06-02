import csv
import datetime
import re

# 处理1957年到2003年的数据
for year in range(1957, 2004):
    filename = 'dst' + str(year)
    # 打开数据文件并读取所有行
    with open(filename, 'r') as f:
        lines = f.readlines()
    # 创建一个空的二维数组
    data = []
    # 初始化时间变量
    current_date = datetime.datetime(year=year, month=1, day=1)
    # 遍历每一行
    for line in lines:
        # 用正则表达式将行分割为多个数字值
        values = re.findall(r'-?\d+', line)
        # 将数字值转换为浮点数并添加到行中
        row = []
        for value in values:
            if value[0] == '-':
                value = value[1:]
                value = int(value)
                row.append(-value)
            else:
                value = int(value)
                row.append(value)
        # 去掉基线值和不需要的数据
        row = row[4:28]
        # 计算平均值并添加到行中
        avg = sum(row) / len(row)
        row.append(avg)
        # 将日期和小时数据添加到二维数组中
        date_str = int(current_date.strftime('%Y%m%d'))
        row = [date_str] + row
        data.append(row)
        # 递增时间变量
        current_date += datetime.timedelta(days=1)
    # 将数据写入到CSV文件中
    with open(filename + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

# 处理2004年及以后的数据
for year in range(2004, 2018):
    filename = 'dst' + str(year)

    # 打开数据文件并读取所有行
    with open(filename, 'r') as f:
        lines = f.readlines()

    # 创建一个空的二维数组
    data = []

    # 初始化时间变量
    current_date = datetime.datetime(year=year, month=1, day=1)

    # 遍历每一行
    for line in lines:
        # 将行分割为多个单元格，并转换为正确的数据类型
        row = []
        for x in line.split():
            if x.isdigit() or ('.' in x and x.replace('.', '', 1).isdigit()):
                row.append(int(x))
            else:
                row.append(x)

        # 提取第一至第二十四小时的数据
        hourly_data = row[2:27]

        # 将日期和小时数据添加到二维数组中
        date_str = current_date.strftime('%Y%m%d')
        row = [date_str] + hourly_data
        data.append(row)

        # 递增时间变量
        current_date += datetime.timedelta(days=1)

    # 将数据写入到CSV文件中
    with open(filename + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)
