import os

xml_path = "/home/jt1/data/train_data/merged_JT002_JT003_bg1_bg2_JT004/xml"
test_txt_path = "/home/jt1/data/train_data/merged_JT002_JT003_bg1_bg2_JT004/test.txt"
train_txt_path = "/home/jt1/data/train_data/merged_JT002_JT003_bg1_bg2_JT004/train.txt"

if __name__ == "__main__":
    with open(train_txt_path, 'w') as wf:
        with open(test_txt_path, 'r') as rf:
            test_lines = [t.replace('\n', '') for t in rf.readlines()]
            print(test_lines)
            for xml in os.listdir(xml_path):
                xml_name, extention = os.path.splitext(xml)
                if xml_name not in test_lines:
                    wf.write(xml_name + '\n')
