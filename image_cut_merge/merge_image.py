import cv2
import numpy as np
import random
import string
import os
import matplotlib.pyplot as plt
import pandas as pd
import argparse

BASE_DIR = os.getcwd()

#-----------------------------------------
# 두 개의 이미지 비교 함수
def compare_two(img1, img2):
    # 비교할 두 개의 이미지의 상 하 좌 우 각 한줄 씩 픽셀 값 비교
    img1_L, img1_R, img1_U, img1_B = img1[:,:1], img1[:, -1:], img1[:1,:], img1[-1:,:]
    img2_L, img2_R, img2_U, img2_B = img2[:,:1], img2[:, -1:], img2[:1,:], img2[-1:,:]

    img1_L_mean = np.mean(np.average(img1_L, axis=(0,1)))
    img1_R_mean = np.mean(np.average(img1_R, axis=(0,1)))
    img1_U_mean = np.mean(np.average(img1_U, axis=(0,1)))
    img1_B_mean = np.mean(np.average(img1_B, axis=(0,1)))
    
    img2_L_mean = np.mean(np.average(img2_L, axis=(0,1)))
    img2_R_mean = np.mean(np.average(img2_R, axis=(0,1)))
    img2_U_mean = np.mean(np.average(img2_U, axis=(0,1)))
    img2_B_mean = np.mean(np.average(img2_B, axis=(0,1)))

    LL = abs(img1_L_mean -  img2_L_mean)
    LR = abs(img1_L_mean -  img2_R_mean)
    RL = abs(img1_R_mean -  img2_L_mean)
    RR = abs(img1_R_mean -  img2_R_mean)
    UU = abs(img1_U_mean -  img2_U_mean)
    UB = abs(img1_U_mean -  img2_B_mean)
    BU = abs(img1_B_mean -  img2_U_mean)
    BB = abs(img1_B_mean -  img2_B_mean)

    min_dir= min(LL,RL,LR,RR,BB,BU,UB,UU)
    ans = {}

    if min_dir == LL :
        ans.update({'LL' : LL})
    elif min_dir == LR :
        ans.update({'LR' : LR})
    elif min_dir == RL :
        ans.update({'RL' : RL})
    elif min_dir == RR :
        ans.update({'RR' : RR})
    elif min_dir== UU :
       ans.update({'UU' : UU})
    elif min_dir == UB :
        ans.update({'UB' : UB})
    elif min_dir == BU :
        ans.update({'BU' : BU})
    elif min_dir == BB :
        ans.update({'BB' : BB})
    return ans


#-----------------------------------------
# 
def make_compare_df(merge_list, first_img):
    compare_df = pd.DataFrame(index=range(0,1),columns=['name','relation','difference'])

    for i in range(len(merge_list)):
        compare_img = cv2.imread(os.path.join(BASE_DIR,'cuted_img',merge_list[i]), cv2.IMREAD_COLOR)
        
        # 비교할 두 개의 이미지 사이즈가 동일하지 않으면 두번째 이미지를 90도 시계방향 회전
        if first_img.shape[0] != compare_img.shape[0] :
            compare_img = cv2.rotate(compare_img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(os.path.join(BASE_DIR,'cuted_img',merge_list[i]), compare_img) #회전 이미지를 동일한 파일이름으로 저장
        
        compare_value = compare_two(first_img, compare_img)
        for key, value in compare_value.items():
            compare_df.loc[i] = [merge_list[i],key,value]
    

    compare_df = compare_df.sort_values(['difference'])
    compare_df['location'] ='diagonal'
    compare_df = compare_df.reset_index(drop=True)

    """
    설명 
    여기에다 적어줭.
    설명이 더 필요하면 주석 달아서 더 붙여줘.
    """
    for j in range(len(compare_df)-1):
        relation=compare_df.iloc[j]['relation']
        name=compare_df.iloc[j]['name']
        compare_img1  = cv2.imread(os.path.join(BASE_DIR,'cuted_img',name), cv2.IMREAD_COLOR)
        
        if (relation == 'LL') | (relation == 'RR'):
            compare_img1 = cv2.flip(compare_img1,1)#좌우반전
            cv2.imwrite(os.path.join(BASE_DIR,'cuted_img',name), compare_img1)
        elif (relation == 'UU') | (relation == 'BB'):  
            compare_img1 = cv2.flip(compare_img1,0)#상하반전
            cv2.imwrite(os.path.join(BASE_DIR,'cuted_img',name), compare_img1)   
        
    
        if (relation == 'LL') | (relation == 'LR'):
            compare_df.loc[j,'location'] = 'left'
        elif (relation == 'RL') | (relation == 'RR'):
            compare_df.loc[j,'location'] = 'right'
        elif (relation == 'UU') | (relation == 'UB'):
            compare_df.loc[j,'location'] = 'upper'
        elif (relation == 'BU') | (relation == 'BB'):
            compare_df.loc[j,'location'] = 'bottom'

    return compare_df

#-----------------------------------------
# 
def merge_img(prefix_name, column_num, row_num, output_name):
    
    merge_list = []
    for patch in os.listdir(os.path.join(BASE_DIR, 'cuted_img')) :
        if patch.split('-')[0] == prefix_name:
            merge_list.append(patch)

    random.shuffle(merge_list)
    print("조각 이미지 리스트: ", merge_list)
    
    if column_num!=2 or row_num!=2:
        print("아직 구현을 못했습니다..column_num=2, row_num=2를 입력해주세요")
    else:
        # 비교할 첫번째 이미지
        first_img = cv2.imread(os.path.join(BASE_DIR,'cuted_img',merge_list.pop(0)), cv2.IMREAD_COLOR)


        first_img_compare_df = make_compare_df(merge_list, first_img)


        # 기준 이미지에서 가로로 붙여지는 이미지들(좌우)
        w = first_img_compare_df[(first_img_compare_df['location']=='right') | (first_img_compare_df['location']=='left')].index.tolist()[0]
        # 기준 이미지에서 위로 붙여지는 이미지들(상하)
        h = first_img_compare_df[(first_img_compare_df['location']=='upper') | (first_img_compare_df['location']=='bottom')].index.tolist()[0]

        #가로, 위, 대각선 이미지 불러오기
        width_img = cv2.imread(os.path.join(BASE_DIR,'cuted_img',first_img_compare_df['name'][w]), cv2.IMREAD_COLOR)
        height_img = cv2.imread(os.path.join(BASE_DIR,'cuted_img',first_img_compare_df['name'][h]), cv2.IMREAD_COLOR)
        diagonal_img = cv2.imread(os.path.join(BASE_DIR,'cuted_img',first_img_compare_df['name'][2]), cv2.IMREAD_COLOR)



        """
        여기는 좀 빡시게 설명... 
        여긴 진짜 이해하기 어려웠엉...
        """
        width_diagonal_value = compare_two(width_img, diagonal_img)
        for key, value in width_diagonal_value.items():
            if first_img_compare_df['location'][h] == 'bottom':
                if (key == "UU") | (key == "UB"):
                    width_img = cv2.flip(width_img,0)
                    cv2.imwrite(os.path.join(BASE_DIR,'cuted_img',first_img_compare_df['name'][w]), width_img)
                if (key == "UB") | (key == "BB"):
                    diagonal_img = cv2.flip(diagonal_img,0)
                    cv2.imwrite(os.path.join(BASE_DIR,'cuted_img',first_img_compare_df['name'][2]), diagonal_img)
            else :
                if (key == "BU") | (key == "BB"):
                    width_img = cv2.flip(width_img,0)
                    cv2.imwrite(os.path.join(BASE_DIR,'cuted_img',first_img_compare_df['name'][w]), width_img)
                if (key == "UU") | (key == "BU"):
                    diagonal_img = cv2.flip(diagonal_img,0)
                    cv2.imwrite(os.path.join(BASE_DIR,'cuted_img',first_img_compare_df['name'][2]), diagonal_img)

        heigth_diagonal_value = compare_two(height_img, diagonal_img)
        for key, value in heigth_diagonal_value.items():
            if first_img_compare_df['location'][w] == 'right' :
                if (key == "LL") | (key == "LR"):
                    height_img = cv2.flip(height_img,1)
                    cv2.imwrite(os.path.join(BASE_DIR,'cuted_img',first_img_compare_df['name'][h]), height_img)
                if (key == "RR") | (key == "LR"):
                    diagonal_img = cv2.flip(diagonal_img,1)
                    cv2.imwrite(os.path.join(BASE_DIR,'cuted_img',first_img_compare_df['name'][2]), diagonal_img)
            else :
                if (key == "RR") | (key == "RL"):
                    height_img = cv2.flip(height_img,1)
                    cv2.imwrite(os.path.join(BASE_DIR,'cuted_img',first_img_compare_df['name'][h]), height_img)
                if (key == "LL") | (key == "RL"):
                    diagonal_img = cv2.flip(diagonal_img,1)
                    cv2.imwrite(os.path.join(BASE_DIR,'cuted_img',first_img_compare_df['name'][2]), diagonal_img)



    
        if first_img_compare_df['location'][w] == 'right' :
            concat_w1 = cv2.hconcat([first_img,width_img])
            concat_w2 = cv2.hconcat([height_img,diagonal_img])
        else :
            concat_w1 = cv2.hconcat([width_img,first_img,])
            concat_w2 = cv2.hconcat([diagonal_img,height_img])

        if first_img_compare_df['location'][h] == 'bottom' :
            concat = cv2.vconcat([concat_w1,concat_w2])
        else :
            concat = cv2.vconcat([concat_w2,concat_w1])
        
        
        # 최종 저장
        concat = cv2.cvtColor(concat, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(BASE_DIR,f'{output_name}.png'), concat)


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_filename_prefix", type=str)
    parser.add_argument("column_num", type=int)
    parser.add_argument("row_num", type=int)
    parser.add_argument("output_filename", type=str)

    configs = parser.parse_args()

    merge_img(configs.input_filename_prefix, configs.column_num, configs.row_num, configs.output_filename)