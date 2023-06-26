import random
import string
import cv2
import os
import argparse

BASE_DIR = os.getcwd()

def cut_img(image, column_num, row_num, prefix_name): 
    """
    image: 이미지 경로
    column_num: 가로로 자를 갯수
    row_num: 세로로 자를 갯수
    prefix_name: 고정된 이름
    """
    if os.path.isdir(os.path.join(BASE_DIR, 'cuted_img')):
        for file in os.scandir(os.path.join(BASE_DIR, 'cuted_img')):
            os.remove(file.path)
    else:
        os.mkdir(os.path.join(BASE_DIR, 'cuted_img'))
    
    h, w, _ = image.shape
    if h % row_num != 0 or w % column_num != 0:
        image = image[:h-(h % row_num), :w-(w % column_num)]

    for i in range(row_num):
        for j in range(column_num):
            cut_img = image[i*int(h//row_num):(i+1)*int(h//row_num), j*int(w//column_num):(j+1)*int(w//column_num)]

            # random 수정
            rand = random.uniform(0,1)
            if rand > 0.5:
                cut_img = cv2.flip(cut_img,1) # 좌우 반전

            rand = random.uniform(0,1)
            if rand > 0.5: 
                cut_img = cv2.flip(cut_img,0) # 상하 반전

            rand = random.uniform(0,1)
            if rand > 0.5:
                cut_img = cv2.rotate(cut_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            random_name_len = 10
            random_name = ""
            for _ in range(random_name_len):
                random_name += str(random.choice(string.ascii_lowercase))
            
            cv2.imwrite(f'cuted_img/{prefix_name}-{random_name}.png', cut_img)



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Divide image into patches")

    parser.add_argument("image_file_name", type=str)
    parser.add_argument("column_num", type=int)
    parser.add_argument("row_num", type=int)
    parser.add_argument("prefix_output_filename", type=str)

    configs = parser.parse_args()

    img = cv2.imread(configs.image_file_name, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cut_img(img, configs.column_num, configs.row_num, configs.prefix_output_filename)
