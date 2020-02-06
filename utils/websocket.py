import socket
import json
from utils import logger


def websocket_server(server_port, q):
    host = socket.gethostbyname(socket.gethostname())
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, server_port))
    server.listen(1)

    client = None
    while True:
        if client is None:
            logger.info('waiting client to connect...')
            client, address = server.accept()
            logger.info(f'client {address} connected!')
        try:
            while not q.empty():
                msg_json = q.get(1)
                client.sendall(msg_json.encode('utf-8'))
        except Exception as e:
            logger.info('client lost...')
            client = None
            logger.error(e)


def creat_json_msg(camera_index, timestamp, frame_index, rects):
    position_json = creat_position_json(rects)
    msg = {'camera_index': camera_index,
           'timestamp': timestamp,
           'frame_index': frame_index,
           'position': position_json}
    msg_json = json.dumps(msg)
    return msg_json


def creat_position_json(rects):
    """
    :param rects: [(x,y,w,h),(x,y,w,h)]
    :return:json
    """
    position = []
    for rect in rects:
        position.append({'x': rect[0], 'y': rect[1], 'w': rect[2], 'h': rect[3]})
    position_json = json.dumps(position)
    return position_json
