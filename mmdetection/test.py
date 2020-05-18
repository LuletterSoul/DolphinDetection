from pathlib import Path

import cv2

from mmdet.apis import init_detector, inference_detector
import mmcv
import os
import imutils

config_file = './configs/cascade_rcnn/cascade_rcnn_r50_caffe_fpn_1x_coco.py'
checkpoint_file = './work_dirs/cascade_rcnn_r50_caffe_fpn_1x_coco/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = '/home/jt1/shw/input_image/'  # or img = mmcv.imread(img), which will only load it once

# for n in os.listdir(img):

#    result = inference_detector(model, os.path.join(img,n))
#    print(result)
#    # visualize the results in a new window
#    model.show_result(os.path.join(img,n), result)
# or save the visualization results to image files
#    model.show_result(os.path.join(img,n), result, out_file=os.path.join("/home/jt1/shw/output_image",n))

# test a video and show the results
# video = mmcv.VideoReader('/home/jt1/Desktop/background_data/0405150859_16.mp4')
# video = mmcv.VideoReader('/home/jt1/Desktop/background_data2/bird/0422141312_10.mp4')
# video = mmcv.VideoReader('/home/jt1/Desktop/background_data2/bird/0422142355_11.mp4')
# video = mmcv.VideoReader('/home/jt1/Desktop/background_data2/pure-bg/0421141556_44.mp4')
# video = mmcv.VideoReader('/home/jt1/Desktop/background_data2/pure-bg/0419105223_31.mp4')
# video = mmcv.VideoReader('/home/jt1/Desktop/background_data2/ship/0420142225_5.mp4')
# video = mmcv.VideoReader('/home/jt1/Desktop/background_data2/ship/0421140514_39.mp4')
video = mmcv.VideoReader('/home/jt1/Desktop/background_data2/ship/0422120614_4.mp4')
# video = mmcv.VideoReader('/home/jt1/Desktop/DolphinDataset/Dolphin/2020-03-03/0303164655_75.mp4')
# video = mmcv.VideoReader('/home/jt1/Desktop/DolphinDataset/Dolphin/2020-04-22/0424144741_3.mp4')
# video = mmcv.VideoReader('/home/jt1/Desktop/DolphinDataset/Dolphin/2020-04-28/0430115154_0.mp4')
# video = mmcv.VideoReader('/home/jt1/Desktop/DolphinDataset/Dolphin/2020-04-28/0430110052_0.mp4')

# video_path = '/home/jt1/Desktop/DolphinDataset/Dolphin/DolphinClassification'
# video_path = '/home/jt1/Desktop/2020-05-15-test/0408'
# video_path = '/home/jt1/Downloads/20200223'
video_path = "/home/jt1/Desktop/DolphinDataset/Dolphin/DolphinClassification"
idx = 0
for vp in os.listdir(video_path):
   for cp in os.listdir(os.path.join(video_path, vp)):
       video = mmcv.VideoReader(os.path.join(video_path, vp, cp))
       for frame in video:
           if idx % 2 == 0:
               frame = imutils.resize(frame,width=1000)
               result = inference_detector(model, frame)
               model.show_result(frame, result, wait_time=1, show=True, font_scale=1, score_thr=0.7)
           idx += 1

# for vp in os.listdir(video_path):
#         video = mmcv.VideoReader(os.path.join(video_path, vp))
#         for frame in video:
#             frame = imutils.resize(frame,width=1000)
#             if idx % 2 == 0:
#                 result = inference_detector(model, frame)
#                 model.show_result(frame, result, wait_time=1, show=True, font_scale=1, score_thr=0.8)
#             idx += 1

# obj_type = 'bird_split'
# obj_path = f'/home/jt1/Desktop/background_data2/split_videos/{obj_type}'
# obj_img_dir = Path(obj_path)
# output_dir = f'/home/jt1/Desktop/background_data2/{obj_type}_filtered'
# if not os.path.exists(output_dir):
#     os.mkdir(output_dir)
# img_paths = list(obj_img_dir.glob('*/*'))
#
# for idx, p in enumerate(img_paths):
#     result = inference_detector(model, str(p))
#     print(f'Proccessed [{idx}/{len(img_paths)}]: {str(p)}')
#     if len(result[0]):
#         if result[0][0][4] > 0.1:
#             print(result[0][0])
#             wp = os.path.join(output_dir, os.path.basename(str(p)))
#             cv2.imwrite(wp,cv2.imread(str(p)))
#             print(f'Write into [{idx}/{len(img_paths)}]: {wp}')
