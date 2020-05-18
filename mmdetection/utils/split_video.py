import time
import os
import cv2 as cv


def open_frame(video_path, img_dir):
    """
    video -> img_dir
    :param video_path:
    :param img_dir:
    :return:
    """
    print(f'--------------------------------------------------------------------------------------')
    print(f'processing [{video_path}]')
    print(f'images save to [{img_dir}]')
    print(f'--------------------------------------------------------------------------------------')
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    video_capture = cv.VideoCapture(video_path)
    idx = 0
    ret, frame = video_capture.read()
    while ret:
        img_path = os.path.join(img_dir, f'{video_base_name}_{idx}.jpg')
        cv.imwrite(img_path, frame)
        print(f'idx={idx}, [{img_path}] saved ...')
        ret, frame = video_capture.read()
        idx += 1
    video_capture.release()


def process1():
    video_dir = 'D:/Datasets/Dolphin/label_work/label_wave/videos'
    output_dir = 'D:/Datasets/Dolphin/label_work/label_wave/images'
    label_dir = 'D:/Datasets/Dolphin/label_work/label_wave/label'
    video_list = os.listdir(video_dir)
    video_list.sort()
    for i in range(len(video_list)):
        print(f'i={i}, name={video_list[i]}')
        video_name = os.path.splitext(video_list[i])[0]
        img_dir = os.path.join(output_dir, f'{video_name}')
        label_dir_ = os.path.join(label_dir, f'{video_name}')
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        if not os.path.exists(label_dir_):
            os.mkdir(label_dir_)
        video_path = os.path.join(video_dir, video_list[i])
        open_frame(video_path, img_dir)


def process2():
    obj_class= 'ship'
    video_dir = f'/home/jt1/Desktop/background_data2/{obj_class}'
    output_dir = f'/home/jt1/Desktop/background_data2/{obj_class}_split'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    video_list = os.listdir(video_dir)
    video_list.sort()
    for i in range(len(video_list)):
        print(f'i={i}, name={video_list[i]}')
        video_name = os.path.splitext(video_list[i])[0]
        img_dir = os.path.join(output_dir, f'{video_name}')
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        video_path = os.path.join(video_dir, video_list[i])
        open_frame(video_path, img_dir)


def main():
    # process1()
    process2()


if __name__ == '__main__':
    main()
