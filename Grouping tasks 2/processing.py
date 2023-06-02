import pandas as pd
import numpy as np

def process_data(data, sensor_id):
    # 处理缺失值
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)

    # 处理异常值
    for col in data.columns:
        if col not in ['ID', '传感器类型']:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            data[col] = np.where((data[col] < lower_bound) | (data[col] > upper_bound), data[col].median(), data[col])

    return data

def ewma_smooth(data, window=3):
    smoothed_data = data.rolling(window=window).mean().dropna()
    # 保留最多两位小数
    smoothed_data = smoothed_data.round(2)
    return smoothed_data
