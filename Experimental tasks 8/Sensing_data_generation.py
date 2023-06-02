import socket
import json
import time
import random

socket_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
socket_send.bind((socket.gethostbyname(socket.gethostname()), 8080))
addr = (socket.gethostbyname(socket.gethostname()), 8000)

data = {
    'Temp': 25,
    'Humi': 88,
    'location': (30, 40)
}

sensor_locations = [(10, 20), (30, 40), (50, 60)]

while True:
    for location in sensor_locations:
        time.sleep(1)
        print('Sending...')
        data['Temp'] = int(random.random() * 10) + 20
        data['Humi'] = int(random.random() * 60) + 20
        if random.random() > 0.75:
            data['Humi'] = None
        if random.random() > 0.75:
            data['Temp'] = None
        data['location'] = location
        socket_send.sendto(bytes(json.dumps(data), encoding='utf-8'), addr)