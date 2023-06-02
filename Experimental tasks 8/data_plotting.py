import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_data(data):
    # 绘图
    plt.figure(figsize=(12,6))
    for name, group in data.groupby(['x', 'y']):
        x, y = name
        plt.plot(group['Temp'], label=f'Temp {(x, y)}')
        plt.plot(group['Humi'], label=f'Humi {(x, y)}')
    plt.xlabel('时间')
    plt.ylabel('数值')
    plt.title('传感器数据')
    plt.legend()
    plt.show()