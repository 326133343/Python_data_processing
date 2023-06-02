import csv 
import datetime 
import statistics 
import re 

# 定义一个函数来计算每个月的均值
def calculate_monthly_average(data):
    monthly_average = []
    for month in range(1, 13):
        monthly_data = [row[month] for row in data if len(row) > month]
        monthly_average.append(round(statistics.mean(monthly_data), 2) if monthly_data else 0)
    return monthly_average

# 创建一个空的二维数组，用于存储所有年份的数据
all_data = []

# 处理1957年到2003年的数据
for year in range(1957, 2004):
    filename = 'dst' + str(year) 
    with open(filename, 'r') as f:
        lines = f.readlines() 
    data = []
    current_date = datetime.datetime(year=year, month=1, day=1) 
    for line in lines:
        values = re.findall(r'-?\d+', line)
        row = []
        for value in values:
            if value[0] == '-':
                value = value[1:]
                value = int(value)
                row.append(-value)
            else:
                value = int(value)
                row.append(value)
        row = row[4:28] 
        avg = sum(row) / len(row) 
        row.append(avg) 
        date_str = int(current_date.strftime('%Y%m%d')) 
        row = [date_str] + row 
        data.append(row) 
        current_date += datetime.timedelta(days=1)
    
    # 计算每个月的均值
    monthly_average = calculate_monthly_average(data)
    
    # 将年份和每个月的均值添加到all_data中
    all_data.append([year] + monthly_average)

# 处理2004年及以后的数据
for year in range(2004, 2018):
    filename = 'dst' + str(year) 
    with open(filename, 'r') as f:
        lines = f.readlines() 
    data = []
    current_date = datetime.datetime(year=year, month=1, day=1) 
    for line in lines:
        row = []
        for x in line.split():
            if x.isdigit() or ('.' in x and x.replace('.', '', 1).isdigit()):
                row.append(int(x))
        hourly_data = row[2:27]
        date_str = current_date.strftime('%Y%m%d') 
        row = [date_str] + hourly_data 
        data.append(row) 
        current_date += datetime.timedelta(days=1)
    
    # 计算每个月的均值
    monthly_average = calculate_monthly_average(data)
    
    # 将年份和每个月的均值添加到all_data中
    all_data.append([year] + monthly_average)

# 将所有年份的数据写入到CSV文件中
with open('all_data_monthly_average.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    writer.writerows(all_data)