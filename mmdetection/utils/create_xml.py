# ! /usr/bin/python

import os, sys
import glob
from PIL import Image

# ICDAR image path
#src_img_dir = "D:/Datasets/Dolphin/data/train_data/SSD_train_data/background_data_picked1/JPEGImages"
#src_ann_dir = "D:/Datasets/Dolphin/data/train_data/SSD_train_data/background_data_picked1/Annotations"
src_img_dir = "/home/jt1/data/train_data/XiaGuan/JPEGImages"
src_ann_dir = "/home/jt1/data/train_data/XiaGuan/Annotations"
os.makedirs(src_ann_dir, exist_ok=True)
img_Lists = glob.glob(src_img_dir + '/*.jpg')

img_name_list = os.listdir(src_img_dir)

for img_name in img_name_list:
    print(f'processing {img_name}')
    img_basename = os.path.splitext(img_name)[0]
    im = Image.open((src_img_dir + '/' + img_basename + '.jpg'))
    width, height = im.size

    # write in xml file
    # os.mknod(src_txt_dir + '/' + img + '.xml')
    xml_path = os.path.join(src_ann_dir, f'{img_basename}.xml')
    xml_file = open(xml_path, 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('\t<folder>simple</folder>\n')
    xml_file.write('\t<filename>' + str(img_basename) + '.jpg' + '</filename>\n')
    xml_file.write('\t<source>\n')
    xml_file.write('\t\t<database>Unknown</database>\n')
    xml_file.write('\t</source>\n')
    xml_file.write('\t<size>\n')
    xml_file.write('\t\t<width>' + str(width) + '</width>\n')
    xml_file.write('\t\t<height>' + str(height) + '</height>\n')
    xml_file.write('\t\t<depth>3</depth>\n')
    xml_file.write('\t</size>\n')
    xml_file.write('\t<segmented>0</segmented>\n')
    xml_file.write('</annotation>')
    xml_file.close()
    print(f'create [{xml_path}]')
