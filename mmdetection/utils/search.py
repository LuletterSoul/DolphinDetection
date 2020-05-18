import os
import shutil


def find_ori_path(labeled_name):
    base_dir = '/home/jt1/deployment/DolphinDetection/data/candidates'
    # 0423/17/original-streams
    date = labeled_name[0:4]
    base_dir = os.path.join(base_dir, date)
    channel_list = os.listdir(base_dir)
    for channel in channel_list:
        temp = os.path.join(base_dir, channel)
        ori_dir = os.path.join(temp, 'original-streams')
        ori_name_list = os.listdir(ori_dir)
        for ori_name in ori_name_list:
            if labeled_name == ori_name:
                return os.path.join(ori_dir, ori_name)
    return None


def pick_ori_video(labeled_dir, output_dir):
    labeled_name_list = os.listdir(labeled_dir)
    for labeled_name in labeled_name_list:
        print(f'processing [{os.path.join(labeled_dir, labeled_name)}]')
        ori_path = find_ori_path(labeled_name)
        if ori_path is not None:
            output_path = os.path.join(output_dir, labeled_name)
            shutil.copy(src=ori_path, dst=output_path)
            print(f'copy [{ori_path}] to [{output_path}]')
        else:
            print('can not find matched original video')


def main():
    # output_dir_1 = '/home/jt1/Desktop/DolphinDataset/Dolphin/2020-04-28'
    labeled_dir_1= '/home/jt1/Desktop/DolphinDataset/Dolphin/2020-04-22'
    output_dir_1 = '/home/jt1/Desktop/0422-original'
    if not os.path.exists(output_dir_1):
        os.mkdir(output_dir_1)

    # labeled_dir_2 = '/home/jt1/Desktop/Datasets/Dolphins/2020-04-28'
    pick_ori_video(labeled_dir_1, output_dir_1)
    # pick_ori_video(labeled_dir_2, output_dir_2)


if __name__ == '__main__':
    main()
