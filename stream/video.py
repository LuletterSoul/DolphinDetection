# -*- coding: utf-8 -*-

import requests
import time
from config import *
from utils.log import logger
import cv2
import queue
from multiprocessing import Queue


class StreamReceiver(object):

    def __init__(self, stream_save_path: Path, offline_path: Path, cfg: VideoConfig, mq: Queue) -> None:
        super().__init__()
        self.stream_save_path = stream_save_path
        self.offline_path = offline_path
        self.cfg = cfg
        self.mq = mq

    def receive_online(self):
        # ts_localpath = "TS"  # 下载的视频段存放路径
        # if not os.path.exists(ts_localpath):
        # os.makedirs(ts_localpath)
        # m3u8_url = "https://222.190.243.176:8081/proxy/video_ts/live/cHZnNjcxLWF2LzE2LzE%3D.m3u8"
        self.index_pool = queue.Queue(self.cfg.max_streams_cache)
        logger.info('Running video [{}] stream receiver process.....'.format(self.cfg.index))
        if self.stream_save_path is None:
            raise Exception('Stream save path cannot be None.')
        if self.cfg is None:
            raise Exception('Video configuration cannot be None.')
        if self.mq is None:
            raise Exception('Multi-processing queue cannot be None.')
        # if not isinstance(mq, ):
        #     raise Exception('Queue must be capable of multi-processing safety.')
        self.stream_save_path.mkdir(exist_ok=True, parents=True)

        pre_index = -1
        # index = []
        # limit max cacheable streams in case storage explosion
        try_cnt = 0
        while True:
            # if len(index) == 0:
            # content = requests.get(m3u8_url).text
            content = requests.get(self.cfg.m3u8_url).text
            lines = content.split('\n')
            for line in lines:
                # print(line)
                if line.endswith(".ts"):
                    # logger.info(line)
                    # i = line.replace("cHZnNjcxLWF2LzE2LzE=/", "").replace(".ts", "")
                    i = line.replace(self.cfg.suffix, "").replace(".ts", "")
                    if int(i) > pre_index or int(i) == 0:
                        self.index_pool.put(int(i))
                        pre_index = int(i)
                        # index.append(int(i))

            # if len(index) == 0:
            #     time.sleep(1)
            #     continue

            # avoid video stream sever lost response if HTTP was frequent
            if self.index_pool.empty():
                try_cnt += 1
                if try_cnt % 20 == 0:
                    logger.info(
                        'Video [{}]: Empty index response from online video. May please connect to video server to fix this network error '
                        'or check the deployment server network connected.'.format(
                            self.cfg.index))
                time.sleep(1)
                continue
            # n = index[0]
            # index.remove(index[0])
            current_index = self.index_pool.get()

            # url = "https://222.190.243.176:8081/proxy/video_ts/live/cHZnNjcxLWF2LzE2LzE=/" + str(n) + ".ts"
            # url = vcfg[URL] + str(n) + ".ts"
            url = self.cfg.url + str(current_index) + ".ts"

            # headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36"}
            # headers = {"User-Agent": "AppleCoreMedia/1.0.0.19B88(Macintosh;U;Intel Mac OS X 10_15_1;en_us)"}
            logger.debug('Send Request: [{}]'.format(url))
            response = requests.get(url, headers=self.cfg.headers)
            logger.debug('Reponse status code: [{}]'.format(response.status_code))

            if response.status_code == 404:
                logger.debug('Stream Not found: [{}]'.format(response.status_code))
                continue

            # f = open(ts_localpath + "/" + "%03d.ts" % n, "wb")
            format_index = str("%03d.ts" % current_index)
            f = open(self.stream_save_path / format_index, "wb")
            f.write(response.content)
            f.close()
            # caches the stream index which has been completed by HTTP request.
            # note that the Queue is p
            self.mq.put(format_index)
            logger.debug("%03d.ts Download~" % current_index)


def read(stream_save_path: Path, cfg: VideoConfig, mq: Queue):
    # ts_localpath = "TS"  # 下载的视频段存放路径
    # if not os.path.exists(ts_localpath):
    # os.makedirs(ts_localpath)
    # m3u8_url = "https://222.190.243.176:8081/proxy/video_ts/live/cHZnNjcxLWF2LzE2LzE%3D.m3u8"
    logger.info('Running video [{}] stream receiver process.....'.format(cfg.index))
    if stream_save_path is None:
        raise Exception('Stream save path cannot be None.')
    if cfg is None:
        raise Exception('Video configuration cannot be None.')
    if mq is None:
        raise Exception('Multi-processing queue cannot be None.')
    # if not isinstance(mq, ):
    #     raise Exception('Queue must be capable of multi-processing safety.')
    stream_save_path.mkdir(exist_ok=True, parents=True)

    pre_index = -1
    # index = []
    q = queue.Queue()
    while True:
        # if len(index) == 0:
        # content = requests.get(m3u8_url).text
        content = requests.get(cfg.m3u8_url).text
        lines = content.split('\n')
        for line in lines:
            # print(line)
            if line.endswith(".ts"):
                # logger.info(line)
                # i = line.replace("cHZnNjcxLWF2LzE2LzE=/", "").replace(".ts", "")
                i = line.replace(cfg.suffix, "").replace(".ts", "")
                if int(i) > pre_index or int(i) == 0:
                    q.put(int(i))
                    pre_index = int(i)
                    # index.append(int(i))

        # if len(index) == 0:
        #     time.sleep(1)
        #     continue

        # avoid video stream sever lost response if HTTP was frequent
        if q.empty():
            time.sleep(1)
            continue
        # n = index[0]
        # index.remove(index[0])
        current_index = q.get()

        # url = "https://222.190.243.176:8081/proxy/video_ts/live/cHZnNjcxLWF2LzE2LzE=/" + str(n) + ".ts"
        # url = vcfg[URL] + str(n) + ".ts"
        url = cfg.url + str(current_index) + ".ts"

        # headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36"}
        # headers = {"User-Agent": "AppleCoreMedia/1.0.0.19B88(Macintosh;U;Intel Mac OS X 10_15_1;en_us)"}
        logger.debug('Send Request: [{}]'.format(url))
        response = requests.get(url, headers=cfg.headers)
        logger.debug('Reponse status code: [{}]'.format(response.status_code))

        if response.status_code == 404:
            logger.debug('Stream Not found: [{}]'.format(response.status_code))
            continue

        # f = open(ts_localpath + "/" + "%03d.ts" % n, "wb")
        format_index = str("%03d.ts" % current_index)
        f = open(stream_save_path / format_index, "wb")
        f.write(response.content)
        f.close()
        # caches the stream index which has been completed by HTTP request.
        # note that the Queue is p
        mq.put(format_index)
        logger.debug("%03d.ts Download~" % current_index)

        # if __name__ == '__main__':
        #     read()


def process_video1(input_path):
    cap = cv2.VideoCapture(input_path)
    # num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if not cap.isOpened():
        print("Input path doesn't exist")
    cnt = 0
    count = 0
    while 1:
        ret, frame = cap.read()
        count += 1
        cv2.imshow('test', frame)
        cv2.waitKey(0)
        # cv2.imwrite(os.path.join(outpath, str(count) + ".jpg"), frame)
        # cv2.imwrite(outpath / str(count) / ".jpg", frame)
        if not ret:
            logger.info('Read frames failed from [{}]'.format(input_path))
            break
    return count
