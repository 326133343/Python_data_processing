import socket
import json
import os
import pandas as pd
from processing import process_data, ewma_smooth
import glob
import signal
import sys

my_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
my_socket.bind((socket.gethostbyname(socket.gethostname()), 7999))
print('连接地址：', my_socket)

recv = []
files = os.listdir()
for i in range(100):
    if 'sensor' + str(i) + '.csv' in files:
        recv.append(i)

# 存储原始数据的 DataFrame
raw_data_list = dict()

def signal_handler(sig, frame):
    # 关闭 socket
    my_socket.close()

    # 删除所有 CSV 文件
    csv_files = glob.glob('sensor*.csv')
    for csv_file in csv_files:
        os.remove(csv_file)

    sys.exit(0)

# 设置信号处理程序
signal.signal(signal.SIGINT, signal_handler)
try:
    while True:
        data, addr = my_socket.recvfrom(1024)
        recv_data = json.loads(str(data, encoding='utf-8'))
        print('连接地址：', addr, '节点ID：', recv_data['ID'])

        df = pd.DataFrame(recv_data, index=[0])

        if recv_data['ID'] in recv:
            df.to_csv('sensor' + str(recv_data['ID']) + '.csv', mode='a', index=False, header=False, columns=recv_data.keys())
        else:
            recv.append(recv_data['ID'])
            df.to_csv('sensor' + str(recv_data['ID']) + '.csv', mode='a', index=False, columns=recv_data.keys())
            raw_data_list[recv_data['ID']] = pd.DataFrame(columns=recv_data.keys())

        raw_data_list[recv_data['ID']] = raw_data_list[recv_data['ID']].append(df, ignore_index=True)

        if len(raw_data_list[recv_data['ID']]) < 3:
            if os.path.exists(f'sensor{recv_data["ID"]}_processed.csv'):
                df.drop(columns=['传感器类型']).to_csv(f'sensor{recv_data["ID"]}_processed.csv', mode='a', index=False, header=False)
            else:
                df.drop(columns=['传感器类型']).to_csv(f'sensor{recv_data["ID"]}_processed.csv', mode='a', index=False)
        else:
            processed_data = process_data(raw_data_list[recv_data['ID']].tail(3), recv_data['ID'])
            smoothed_data = ewma_smooth(processed_data)
            if os.path.exists(f'sensor{recv_data["ID"]}_processed.csv'):
                smoothed_data.tail(1).to_csv(f'sensor{recv_data["ID"]}_processed.csv', mode='a', index=False, header=False)
            else:
                smoothed_data.tail(1).to_csv(f'sensor{recv_data["ID"]}_processed.csv', mode='a', index=False)

except KeyboardInterrupt:
    signal_handler(signal.SIGINT,None)