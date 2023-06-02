import socket
import json
import pandas as pd
import time
from data_processing import process_data
from data_plotting import plot_data

def recv_data():
    my_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    my_socket.bind((socket.gethostbyname(socket.gethostname()), 8000)) # 接收端口为8000

    # 创建空的DataFrame
    data = pd.DataFrame()

    try:
        for i in range(100):
            recv_data, addr = my_socket.recvfrom(1024)
            recv_data = json.loads(str(recv_data, encoding='utf-8'))
            # 接收的原始数据
            print(recv_data)
            
            recv_data['x'], recv_data['y'] = recv_data['location']
            recv_data.pop('location')
            # 处理后的数据
            print(recv_data)
            
            ## 将recv_data转换成只有一行的DataFrame类型
            recv_data = pd.DataFrame(recv_data, index=[pd.Timestamp.now()])
            
            ## 将data与recv_data拼接成一个DataFrame
            data = pd.concat([data, recv_data])
    finally:
        # 关闭套接字
        my_socket.close()

        # 保存数据
        data.to_csv('raw_data.csv')
        
        process_data(data)
        plot_data(data)

if __name__ == "__main__":
    recv_data()