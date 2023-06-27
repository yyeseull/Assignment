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

    #두 이미지의 edge의 평균
    img1_L_mean = np.mean(np.average(img1_L, axis=(0,1)))
    img1_R_mean = np.mean(np.average(img1_R, axis=(0,1)))
    img1_U_mean = np.mean(np.average(img1_U, axis=(0,1)))
    img1_B_mean = np.mean(np.average(img1_B, axis=(0,1)))
    
    img2_L_mean = np.mean(np.average(img2_L, axis=(0,1)))
    img2_R_mean = np.mean(np.average(img2_R, axis=(0,1)))
    img2_U_mean = np.mean(np.average(img2_U, axis=(0,1)))
    img2_B_mean = np.mean(np.average(img2_B, axis=(0,1)))


    #두 이미지의 가로, 세로의 차
    LL = abs(img1_L_mean -  img2_L_mean)
    LR = abs(img1_L_mean -  img2_R_mean)
    RL = abs(img1_R_mean -  img2_L_mean)
    RR = abs(img1_R_mean -  img2_R_mean)
    UU = abs(img1_U_mean -  img2_U_mean)
    UB = abs(img1_U_mean -  img2_B_mean)
    BU = abs(img1_B_mean -  img2_U_mean)
    BB = abs(img1_B_mean -  img2_B_mean)

    #평균의 차가 가장 최소한 부분이 유사한 부분일 가능성이 높음.

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
"""
기준 이미지와의 관계를 dataframe을 통해 저장할 함수
- 데이터프레임 컬럼 설명
- [name] : 이미지 파일명
- [relation] : 기준이미지와 유사한 부분 위치
- [difference] : 두 이미지의 차
- [location] : 기준이미지에서 비교이미지의 위치(좌,우,상,하)
"""
def make_compare_df(merge_list, first_img):
    compare_df = pd.DataFrame(index=range(0,1),columns=['name','relation','difference'])

    for i in range(len(merge_list)):
        compare_img = cv2.imread(os.path.join(BASE_DIR,'cuted_img',merge_list[i]), cv2.IMREAD_COLOR)
        
        # 기준 이미지의 크기와 동일하도록 조정. 이미지 크기가 동일하지 않으면 비교 이미지를 90도 시계방향 회전
        if first_img.shape[0] != compare_img.shape[0] :
            compare_img = cv2.rotate(compare_img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(os.path.join(BASE_DIR,'cuted_img',merge_list[i]), compare_img) #회전 이미지를 동일한 파일이름으로 저장
        
        compare_value = compare_two(first_img, compare_img)
        for key, value in compare_value.items():
            compare_df.loc[i] = [merge_list[i],key,value]
    
    # 기준 이미지와 그 외 이미지를 compare_two함수를 통해 'compare_df'데이터프레임에 기록후 정렬 
    compare_df = compare_df.sort_values(['difference'])
    compare_df['location'] ='diagonal' 
    compare_df = compare_df.reset_index(drop=True)

    """
    기준 이미지는 고정한다.
    기준 이미지와 비교하는 이미지의 관계에 따라 비교하는 이미지 좌우상하반전
    예를 들어 두 이미지가 좌우 관계라하면, 
    비슷한 부분이 (오른쪽,왼쪽) 또는 (왼쪽,오른쪽)이 합쳐져야할 부분인데
    두 이미지가 (오른쪽,오른쪽) 또는 (왼쪽,왼쪽)이라고 하면 하나의 이미지는 적어도 좌우반전이 이루워진것이다.
    따라서 기준 이미지는 고정이므로 비교하는 이미지를 좌우반전 시켜준 뒤 다시 저장해준다. 
    상하 이미지도 똑같은 방향으로 진행한다. 

    """
    for j in range(len(compare_df)-1): #유사도 차이가 가장 높은 이미지는 기준이미지와 대각선 관계이므로 제외하고 진행
        relation=compare_df.iloc[j]['relation']
        name=compare_df.iloc[j]['name']
        compare_img1  = cv2.imread(os.path.join(BASE_DIR,'cuted_img',name), cv2.IMREAD_COLOR)
        
        if (relation == 'LL') | (relation == 'RR'):
            compare_img1 = cv2.flip(compare_img1,1)#좌우반전
            cv2.imwrite(os.path.join(BASE_DIR,'cuted_img',name), compare_img1)
        elif (relation == 'UU') | (relation == 'BB'):  
            compare_img1 = cv2.flip(compare_img1,0)#상하반전
            cv2.imwrite(os.path.join(BASE_DIR,'cuted_img',name), compare_img1)   
        
        """
        기준 이미지와 위치를 지정해준다. 
        예를 들어 관계가 'LR또는 LL'이리고 한다면, 
        기준 이미지 왼쪽부분이 비교 이미지와 유사하다는 것이다. 
        따라서 비교하는 이미지는 기준 이미지의 왼쪽에 병합해주어야한다.
        그 값을 [location] 컬럼에 추가해준다.
        다른 방향도 동일한 방법으로 진행한다.
        """
    
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

    random.shuffle(merge_list) # 이미지가 순서대로 저장이 되어서 불러 올떄 순서를 알 수 있으므로 불러오는 순서 랜덤으로 가지고 오기
    print("조각 이미지 리스트: ", merge_list)
    
    if column_num!=2 or row_num!=2:
        print("아직 구현을 못했습니다..column_num=2, row_num=2를 입력해주세요")
    else:
        # 기준이미지.
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
        기준이미지와 가로로 병합할 이미지 -> w이미지, 세로로 병합할 이미지 -> h이미지, 대각선으로 병합할 이미지 ->  d이미지로 부르겠다.
        기준 이미지를 통해 w이미지는 좌우반전, h이미지는 상하반전이 이루워졌다, 
        여기서는 w이미지는 상히빈잔, h이미지는 좌우반전 , d이미지 두가지 다 진행한다.
        예를 들어
        w이미지와 d이미지의 compare_two 함수를 통해 두 이미지의 관계를 확인한다. 
        여기서 h이미지에 따라서 두 이미지의 상하반전이 결정된다.
        예를 들어 h이미지가 기준이미지의 밑에 위치한다면 
        d이미지는 w이미지의 아래에 위치해야한다. 
        그렇다면 w이미지가 d이미지와 결합하는 부분은 바닥부분(b)가 되어야하고, d이미지가 w이미지와 결합할 부분은 윗부분(u)이 결과가 나와야한다
        이 경우가 아닌 경우는 모두 반전을 상하반전을 시켜준다.

        h이미지와 d이미지도 마찬가지로 w이미지 위치에 따라 똑같은 방식으로 진행하여 좌우반전을 시켜준다. 

        """
         #상하반전
        width_diagonal_value = compare_two(width_img, diagonal_img)
        for key, value in width_diagonal_value.items():
            if first_img_compare_df['location'][h] == 'bottom':
                #가로로 병합할 이미지 상하반전
                if (key == "UU") | (key == "UB"):
                    width_img = cv2.flip(width_img,0)
                    cv2.imwrite(os.path.join(BASE_DIR,'cuted_img',first_img_compare_df['name'][w]), width_img)
                #대각선으로 병합할 이미지 상하반전
                if (key == "UB") | (key == "BB"):
                    diagonal_img = cv2.flip(diagonal_img,0)
                    cv2.imwrite(os.path.join(BASE_DIR,'cuted_img',first_img_compare_df['name'][2]), diagonal_img)
            else :
               #가로로 병합할 이미지 상하반전
                if (key == "BU") | (key == "BB"):
                    width_img = cv2.flip(width_img,0)
                    cv2.imwrite(os.path.join(BASE_DIR,'cuted_img',first_img_compare_df['name'][w]), width_img)
                #대각선으로 병합할 이미지 상하반전
                if (key == "UU") | (key == "BU"):
                    diagonal_img = cv2.flip(diagonal_img,0)
                    cv2.imwrite(os.path.join(BASE_DIR,'cuted_img',first_img_compare_df['name'][2]), diagonal_img)
        #좌우반전
        heigth_diagonal_value = compare_two(height_img, diagonal_img)
        for key, value in heigth_diagonal_value.items():
            if first_img_compare_df['location'][w] == 'right' :
                 #세로로 병합할 이미지 좌우반전
                if (key == "LL") | (key == "LR"):
                    height_img = cv2.flip(height_img,1)
                    cv2.imwrite(os.path.join(BASE_DIR,'cuted_img',first_img_compare_df['name'][h]), height_img)
                #대각선으로 병합할 이미지 좌우반전
                if (key == "RR") | (key == "LR"):
                    diagonal_img = cv2.flip(diagonal_img,1)
                    cv2.imwrite(os.path.join(BASE_DIR,'cuted_img',first_img_compare_df['name'][2]), diagonal_img)
            else :
                 #세로로 병합할 이미지 좌우반전
                if (key == "RR") | (key == "RL"):
                    height_img = cv2.flip(height_img,1)
                    cv2.imwrite(os.path.join(BASE_DIR,'cuted_img',first_img_compare_df['name'][h]), height_img)
                 #대각선으로 병합할 이미지 좌우반전
                if (key == "LL") | (key == "RL"):
                    diagonal_img = cv2.flip(diagonal_img,1)
                    cv2.imwrite(os.path.join(BASE_DIR,'cuted_img',first_img_compare_df['name'][2]), diagonal_img)


        #가로로 병합하기 
    
        if first_img_compare_df['location'][w] == 'right' :
            concat_w1 = cv2.hconcat([first_img,width_img])
            concat_w2 = cv2.hconcat([height_img,diagonal_img])
        else :
            concat_w1 = cv2.hconcat([width_img,first_img,])
            concat_w2 = cv2.hconcat([diagonal_img,height_img])

        #가로로 병합한 두 이미지를 세로로 병합하기
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