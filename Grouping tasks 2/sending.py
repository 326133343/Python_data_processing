import socket
import json
import time
import random
import threading

sensor_types = {'T': '温度', 'H': '湿度', 'F': '烟感', 'S': '超声波', 'P': 'PM2.5', 'L': '光照强度', 'B': '气压', 'N': '噪声'}

class sender():
    def __init__(self, Id, Type=None, Period=None):
        self.Id = Id  # 节点ID号
        if Type is None:
            Type = ''.join(random.sample('THFSPLB', random.randint(2, 5)))
        self.type = Type  # 表示传感器类型
        if Period is None:
            Period = random.randint(5, 10)
        self.Period = Period  # 发送周期
        self.port = 8000 + Id  # 端口号
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((socket.gethostbyname(socket.gethostname()), self.port))
        self.dest_addr = (socket.gethostbyname(socket.gethostname()), 7999)  # 将等号后面的替换成(dest_ip,dest_port),如('192.168.1.109',8000)

    def run(self):
        self.data_base = {}
        self.data_range = {}
        for s in self.type:  # 为每一种传感器类型设定一个随机数基准，和波动性范围
            if s == 'T':  # 温度范围
                self.data_base[s] = random.randint(25, 30)
                self.data_range[s] = random.randint(1, 5)
            if s == 'H':  # 湿度范围
                self.data_base[s] = random.randint(50, 80)
                self.data_range[s] = random.randint(3, 10)
            if s == 'F':  # 烟感范围
                self.data_base[s] = random.randint(10, 20)
                self.data_range[s] = random.randint(1, 5)
            if s == 'S':  # 超声波测距范围
                self.data_base[s] = random.randint(50, 100)
                self.data_range[s] = random.randint(3, 10)
            if s == 'P':  # PM2.5范围
                self.data_base[s] = random.randint(10, 250)
                self.data_range[s] = random.randint(10, 20)
            if s == 'L':  # 光照强度范围
                self.data_base[s] = random.randint(100, 1000)
                self.data_range[s] = random.randint(20, 50)
            if s == 'B':  # 气压范围
                self.data_base[s] = random.randint(900, 1100)
                self.data_range[s] = random.randint(5, 15)
            if s == 'N':  # 噪声范围
                self.data_base[s] = random.randint(20, 80)
                self.data_range[s] = random.randint(5, 20)
            
        # 开启多线程发送
        self.sending_thread = threading.Thread(target=self.sending, args=())
        self.sending_thread.start()

    def getdata(self):
        data = {'ID': self.Id, '传感器类型': self.type}
        for s in self.type:
            prob = random.random()  # 随机生成0-1之间的数
            if prob < 0.1:  # 10%的概率发送空值
                data[sensor_types[s]] = None
            elif prob < 0.25:  # 15%的概率发送异常数据
                data[sensor_types[s]] = self.data_base[s] + 20 * random.randint(0, self.data_range[s])
            else:
                data[sensor_types[s]] = self.data_base[s] + random.randint(0, self.data_range[s])

        return data

    def sending(self):
        while True:
            time.sleep(self.Period)
            print('sensor %d is sending' % self.Id)
            self.sock.sendto(bytes(json.dumps(self.getdata()), encoding='utf-8'), self.dest_addr)

if __name__ == '__main__':
    N = 20
    s = []

    for i in range(1, N + 1):
        s.append(sender(Id=i))

    print('init...')
    for i in s:
        i.run()
        time.sleep(random.randint(2,5))