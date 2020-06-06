import json
import socket
import websockets
import asyncio

from config import ServerConfig, VideoConfig
from utils import logger
import os


async def main_logic(q, vcfg: VideoConfig, scfg: ServerConfig):
    address = f'ws://{scfg.wc_ip}:{scfg.wc_port}'
    history_msg_json = None
    msg_json = None
    # if not scfg.send_msg:
    #     logger.info(f'Controller [{vcfg.index}]: Skipped message by server config specifing.')
    #     return
    while True:
        logger.info(f'waiting to connect to server {address}...')
        async with websockets.connect(address) as server:
            logger.info(f'connect to server {address} successfully')
            flag = True
            if not scfg.send_msg:
                logger.info(f'Controller [{vcfg.index}]: Skipped message from server config indication.')
                return
            while flag:
                try:
                    if history_msg_json is not None:
                        await server.send(history_msg_json.encode('utf-8'))
                        logger.info(f'client send history message to server {address} successfully')
                        history_msg_json = None
                        response_str = await server.recv()
                        logger.info(f'response from server: {response_str}')
                    try:
                        if not q.empty():
                            logger.info(f'Controller [{vcfg.index}]: Current message num: {q.qsize()}')
                            msg_json = q.get(1)
                        else:
                            msg_json = None
                    except Exception as e:
                        logger.error(e)
                        return
                    if msg_json is not None:
                        await server.send(msg_json.encode('utf-8'))
                        logger.info(f'client send message to server {address} successfully: {msg_json}')
                        response_str = await server.recv()
                        logger.info(f'response from server: {response_str}')
                except Exception as e:
                    history_msg_json = msg_json
                    logger.error(e)
                    flag = False


def websocket_client(q, vcfg: VideoConfig, scfg: ServerConfig):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main_logic(q=q, vcfg=vcfg, scfg=scfg))


def socket_client(q, vcfg: VideoConfig, scfg: ServerConfig):
    address = (scfg.wc_ip, scfg.wc_port)
    server = None
    history_msg_json = None
    msg_json = None
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
                logger.info(f'Controller [{vcfg.index}]: Current message num: {q.qsize()}')
                msg_json = q.get(1)
                server.send(msg_json.encode('utf-8'))
                logger.info(f'client send message to server {address} successfully: {msg_json}')
                # time.sleep(10)
        except Exception as e:
            server = None
            history_msg_json = msg_json
            logger.error(e)


async def websocket_client_async(q):
    # address = (WEBSOCKET_SERVER_IP, WEBSOCKET_SERVER_PORT)
    url = "ws://localhost:8765"
    server = None
    history_msg_json = None
    try:
        async with websockets.connect(url) as websocket:
            while True:
                if history_msg_json is not None:
                    server.send(history_msg_json.encode('utf-8'))
                    logger.info(f'client send history message to server {url} successfully')
                    history_msg_json = None
                    # while not q.empty():
                    #     print(q.qsize())
                msg_json = q.get()
                await websocket.send(msg_json.encode('utf-8'))
                logger.info(f'client send message to server {url} successfully')
                # time.sleep(10)
    except Exception as e:
        server = None
        history_msg_json = msg_json
        logger.error(e)


def creat_position_json(rects, cfg: VideoConfig):
    """
    :param rects: [(x,y,w,h),(x,y,w,h)]
    :return:json
    Args:
        cfg:
    """
    new_rects = rects.copy()
    if 'real_shape' in cfg.alg:
        # recover original size such as 1080P position back to 4K
        real_shape = cfg.alg['real_shape']
        current_shape = cfg.shape
        ratio = real_shape[0] / current_shape[0]
        new_rects = []
        for rect in rects:
            r = [rect[0] * ratio, rect[1] * ratio, rect[2] * ratio, rect[3] * ratio]
            new_rects.append(r)
    position = []
    for rect in new_rects:
        position.append({'lx': int(rect[0]), 'ly': int(rect[1]), 'rx': int(rect[2]), 'ry': int(rect[3])})
    position_json = json.dumps(position)
    return position_json


def creat_detect_empty_msg_json(video_stream, channel, timestamp, dol_id=10000, camera_id='camera_bp_1'):
    msg = {
        'cmdType': 'notify',
        "appId": "10080",
        'clientId': 'jt001',
        'data': {
            'notifyType': 'detectedNotify',
            'videoStream': video_stream,
            "cameraId": camera_id,
            "channel": channel,
            'jt_id': str(dol_id),
            'timestamp': timestamp,
            'coordinates': [],
        }
    }
    msg_json = json.dumps(msg)
    return msg_json


def creat_detect_msg_json(video_stream, channel, timestamp, rects, dol_id, camera_id, cfg):
    position_json = creat_position_json(rects, cfg)
    msg = {
        'cmdType': 'notify',
        "appId": "10080",
        'clientId': 'jt001',
        'data': {
            'notifyType': 'detectedNotify',
            'videoStream': video_stream,
            "cameraId": camera_id,
            'channel': channel,
            'jt_id': str(dol_id),
            'timestamp': timestamp,
            'coordinates': position_json,
        }
    }
    msg_json = json.dumps(msg)
    return msg_json


def creat_packaged_msg_json(filename, path, cfg: VideoConfig, camera_id, channel, preview_name):
    url = os.path.join(cfg.dip, 'video', cfg.date, str(cfg.index), filename)
    preview = os.path.join(cfg.dip, 'preview', cfg.date, str(cfg.index), preview_name)
    msg = {
        'cmdType': 'notify',
        'clientId': 'jt001',
        'cameraId': camera_id,
        'channel': channel,
        'data': {
            'notifyType': 'packagedNotify',
            'filename': filename,
            'path': path,
            'url': url,
            'preview': preview
        }
    }
    msg_json = json.dumps(msg)
    return msg_json
