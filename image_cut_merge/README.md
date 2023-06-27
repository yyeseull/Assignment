# Image Cut & Merge

## 문제
1. 이미지를 MxN(2x2, 3x3)으로 자른다.(자를때 크기가 맞지 않는다면 임의로 크기를 조정하고 진행)
2. 자른 sub image는 0.5확률로 mirroring,flipping, 90 degree rotation등을 통해 하나의 완성된 이미지로 만들어낸다
3. 사진을 저장할 때 이름으로 위치를 유추할 수 없게 랜덤으로 진행한다.
4. sub image를 mirroring,flipping, 90 degree rotation등을 통해 merge하여 자연스러운 이미지를 완성한다. 

## Cut 
- MxN으로 진행하였음. 
- 자른 이미지는 순서와 위치를 알 수 없도록 랜덤으로 지정해주었다. 

## Merge 결과 
- 2x2 자른 sub image들의 각 edge들 픽셀 한줄의 RGB채널 각각 평균을 통해 RGB 전체  평균을 통해 구하여 sub image 랜덤으로 기준이 되는 이미지를 정한 후 기준이미지에 따라 다른 이미지들을 mirroring,flipping, 90 degree rotation 진행하여 결합.
  

- 원본 이미지
![merge](https://github.com/yyeseull/Assignment/blob/main/image_cut_merge/cut_merge_img.jpeg?raw=true) 
  
- 병합한 이미지 결과 
![merge](https://github.com/yyeseull/Assignment/assets/102211628/c4cc802f-c662-49c4-8be7-8fac12831b44)

## 한계점 
1. RGB채널을 통해 유사한 부분을 찾는 것을 진행하여서 배경이 한 색으로 통일 된 사진이나, 배경이 대칭되어 있는 사진은 아래사진처럼 자연스럽게 merge가 되지 않는 경우가 있음.

![병합이잘안됨](https://github.com/yyeseull/Assignment/assets/102211628/12c64ece-28dd-42d5-b9ae-0d928b20c824)

2. 2x2까지 실행은 되지만 3x3은 가로, 세로 각각 한줄씩 위치를 찾아가면서 구현을 해볼려고 했지만 9조각이 되니깐 색만으로는 구분을 하기가 어렵게 되어서 진행을 하지 못하였다. 

## 해결방안
- 색만을 이용하여 유사도를 찾다보니 배경사진에 영향을 많이 받으므로 외곽선도 함꼐 비교하여 진행하면 좀 더 오류를 줄여 나 갈 수 있으며, 일반화를 될 수 있도록 개선을 해야함.

## main.sh 입력 방법
```sh main.sh image_file_name column_num row_num prefix_file_name output_file_name```

