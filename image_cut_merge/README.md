# Image Cut & Merge

## 문제
1. 이미지를 MxN(2x2, 3x3)으로 자른다.(자를때 크기가 맞지 않는다면 임의로 크기를 조정하고 진행)
2. 자른 sub image는 0.5확률로 mirroring,flipping, 90 degree rotation등을 통해 하나의 완성된 이미지로 만들어낸다
3. 사진을 저장할 때 이름으로 위치를 유추할 수 없게 랜덤으로 진행한다.
4. sub image를 mirroring,flipping, 90 degree rotation등을 통해 merge하여 자연스러운 이미지를 완성한다. 

## Cut 
- MxN으로 진행하였음. 
- 자른 이미지는 순서와 위치를 알 수 없도록 랜덤으로 지정해주었다. 

## Merge 
- 2x2 자른후 

![원본사진](https://github.com/yyeseull/Assignment/blob/main/image_cut_merge/cut_merge_img.jpeg?raw=true)

![merge](https://github.com/yyeseull/Assignment/assets/102211628/c4cc802f-c662-49c4-8be7-8fac12831b44)