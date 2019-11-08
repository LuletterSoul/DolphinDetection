# -*- coding: utf-8 -*-

import requests
import time
import os


if __name__ == '__main__':

    ts_localpath = "TS"     #下载的视频段存放路径
    if not os.path.exists(ts_localpath):
        os.makedirs(ts_localpath)

    m3u8_url = "https://222.190.243.176:8081/proxy/video_ts/live/cHZnNjcxLWF2LzE2LzE%3D.m3u8"

    n = -1
    index = []

    while True:
        if len(index) == 0:
            content = requests.get(m3u8_url).text
            lines = content.split('\n')
            for line in lines:
                # print(line)
                if (line.endswith(".ts")):
                    i = line.replace("cHZnNjcxLWF2LzE2LzE=/", "").replace(".ts", "")
                    if(int(i) > n or int(i) == 0):
                        index.append(int(i))

        if len(index) == 0:
            time.sleep(1)
            continue

        n = index[0]
        index.remove(index[0])
        
        url = "https://222.190.243.176:8081/proxy/video_ts/live/cHZnNjcxLWF2LzE2LzE=/"+str(n)+".ts"

        #headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36"}
        headers = {"User-Agent": "AppleCoreMedia/1.0.0.19B88(Macintosh;U;Intel Mac OS X 10_15_1;en_us)"}
        response = requests.get(url, headers=headers)

        f = open(ts_localpath+"/"+"%03d.ts" % n, "wb")
        f.write(response.content)
        f.close()
        print("%03d.ts Download~" % n)
        

