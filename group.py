import cv2

import os, shutil
from pathlib import Path


def mymovefile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(dstfile)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # 创建路径
        shutil.move(srcfile, dstfile)  # 移动文件


def mycopyfile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        return
    else:
        fpath, fname = os.path.split(dstfile)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # 创建路径
        shutil.copyfile(srcfile, dstfile)  # 复制文件


input_dir = '/home/jt1/baidunetdiskdownload'
output_dir = Path('/home/jt1/group')

output_dir.mkdir(parents=True, exist_ok=True)

folders = os.listdir(input_dir)
classes = os.listdir(f'{input_dir}/{folders[0]}/main')
for c in classes:
    cls, extention = os.path.splitext(c)
    (output_dir / cls).mkdir(parents=True, exist_ok=True)

for f in folders:
    if os.path.exists(f'{input_dir}/{f}/main'):
        classes = os.listdir(f'{input_dir}/{f}/main')
        for c in classes:
            cls, extention = os.path.splitext(c)
            out = f'{output_dir}/{cls}'
            if not os.path.exists(out):
                os.mkdir(out)
            num = len(os.listdir(out))
            out_file = f'{output_dir}/{cls}/{cls}_{num}.mp4'
            mymovefile(f'{input_dir}/{f}/main/{c}', out_file)
