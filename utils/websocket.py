import socket
import json
import time
from utils import logger
from config import WEBSOCKET_SERVER_IP, WEBSOCKET_SERVER_PORT


def websocket_client(q):
    address = (WEBSOCKET_SERVER_IP, WEBSOCKET_SERVER_PORT)
    server = None
    history_msg_json = None
    while True:
        if server is None:
            logger.info(f'waiting to connect to server {address}...')
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.connect(address)
            logger.info(f'connect to server {address} successfully')
        try:
            if history_msg_json is not None:
                server.send(history_msg_json.encode('utf-8'))
                logger.info(f'client send history message to server {address} successfully')
                history_msg_json = None
            while not q.empty():
                msg_json = q.get(1)
                server.send(msg_json.encode('utf-8'))
                logger.info(f'client send message to server {address} successfully')
                # time.sleep(10)
        except Exception as e:
            server = None
            history_msg_json = msg_json
            logger.error(e)


def creat_position_json(rects):
    """
    :param rects: [(x,y,w,h),(x,y,w,h)]
    :return:json
    """
    position = []
    for rect in rects:
        position.append({'lx': rect[0], 'ly': rect[1], 'rx': rect[0] + rect[2], 'ry': rect[1] + rect[3]})
    position_json = json.dumps(position)
    return position_json


def creat_detect_msg_json(video_stream, channel, timestamp, rects):
    position_json = creat_position_json(rects)
    msg = {
        'cmdType': 'notify',
        "appId": "10080",
        'clientId': 'jt001',
        'data': {
            'notifyType': 'detectedNotify',
            'videoStream': video_stream,
            'channel': channel,
            'timestamp': timestamp,
            'coordinates': position_json
        }
    }
    msg_json = json.dumps(msg)
    return msg_json


def creat_packaged_msg_json(filename, path):
    msg = {
        'cmdType': 'notify',
        'clientId': 'jt001',
        'data': {
            'notifyType': 'packagedNotify',
            'filename': filename,
            'path': path
        }
    }
    msg_json = json.dumps(msg)
    return msg_json
