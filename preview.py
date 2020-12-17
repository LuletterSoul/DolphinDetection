import  os
import cv2

with open('/home/jt1/Downloads/preview.txt','r') as f:
    lines = [l.replace('\n','').replace('http://10.196.122.94:8080/preview','/home/jt1/deployment/DolphinDetection/data/candidates') for l in f.readlines()]
    for l in lines:
        dirname = os.path.dirname(l)
        base_name = os.path.basename(l)
        path = f'{dirname}/preview/{base_name}'
        if os.path.exists(path):
            name = os.path.splitext(base_name)[0]
            big_name = f'{name}_big.png'
            preview = cv2.imread(path)
            preview_small = cv2.resize(preview,dsize=(0,0),fx=0.25,fy=0.25)
            #cv2.imshow('preview',preview_small)
            #cv2.waitKey(0)

            big_path = os.path.join(dirname,'preview',big_name)
            print(big_path)
            cv2.imwrite(big_path,preview)
            cv2.imwrite(path,preview_small)

