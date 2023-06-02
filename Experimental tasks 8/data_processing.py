import pandas as pd
import numpy as np

def process_data(data):
    # 处理data中的数据
    ## 将Temp和Humi列中的异常值和缺省值处理掉
    data['Temp'].replace('', np.nan, inplace=True)  # 将缺失值替换为Nan
    data['Humi'].replace('', np.nan, inplace=True)  # 将缺失值替换为Nan

    # 使用前向填充方法填充缺失值
    data.fillna(method='ffill', inplace=True)
    
    # 或者使用线性插值法填充缺失值
    # data.interpolate(method='time', inplace=True)

    ## 完成分组后不同传感器的温湿度均值计算
    grouped = data.groupby(['x', 'y']).mean()
    print(grouped)

    # 保存处理后的数据
    data.to_csv('processed_data.csv')