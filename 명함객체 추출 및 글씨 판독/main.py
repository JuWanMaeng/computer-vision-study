import cv2
import sys
import numpy as np
import random
import pytesseract

def reorderPts(pts):
    idx=np.lexsort((pts[:,1],pts[:,0]))    #칼럼0 -> 칼럼 1 순으로 정렬한 인덱스를 반환  [0,3,1,2]가 나올것임
    pts=pts[idx]   #x좌표로 정렬

    if pts[0,1]> pts[1,1]:
        pts[[0,1]]=pts[[1,0]]    #두 점을 스와핑 하는 코드

    if pts [2,1]<pts[3,1]:
        pts[[2,3]]=pts[[3,2]]

    return pts

#영상 불러오기
filename='namecard.jpg'
if len(sys.argv)>1:
    filename=sys.argv[1]

src=cv2.imread(filename)

if src is None:
    print('Image load failed!')
    sys.exit()

src=cv2.resize(src,(1080,810))

#출력 영상 설정
dw,dh= 1080,720
secQuad=np.array([[0,0],[0,0],[0,0],[0,0]],np.float32)     #입력영상에서 명함부분의 4개의 모서리 부분점을 저장할 ndarray
dstQuad=np.array([[0,0],[0,dh],[dw,dh],[dw,0]],np.float32)
dst=np.zeros((dh,dw),np.uint8)

#입력 영상 전처리
src_gray=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
_,src_bin=cv2.threshold(src_gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#외곽선 검출 및 명함 검출
contours,_=cv2.findContours(src_bin,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  #바깥쪽에 있는 외각선만 검출


cpy=src.copy()
for pts in contours:
    #너무 작은 객체는 무시
    if cv2.contourArea(pts)<100:
        continue

    #외곽선 근사화
    approx=cv2.approxPolyDP(pts,cv2.arcLength(pts,True)*0.02,True)

    #컨벡스가 아니고, 사각형이 아니면 무시 isContourConvex 는 빼도 잘 작동함
    if not cv2.isContourConvex(approx) or len(approx)!=4:
        continue

    cv2.polylines(cpy,[approx],True,(0,255,0),2,cv2.LINE_AA)  #이 코드는 없어도 될듯
    #approx라는 ndarray에 들어가 있는 점들을 분석해서 좌측상단부터 반시계 방향으로 reorder 해주는 함수이다
    #결과값은 좌측 하단부터 시계방향으로 나온다
    srcQuad=reorderPts(approx.reshape(4,2).astype(np.float32))  


pers=cv2.getPerspectiveTransform(srcQuad,dstQuad)   #원본좌표점 --> 결과 좌표점으로 투시 변환행렬을 구함
dst=cv2.warpPerspective(src,pers,(dw,dh))           #위에서 얻은 투시변환 행렬로 결과 영상을 얻음


dst_gray=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)                       #teseract를 이용한 글씨 판독
print(pytesseract.image_to_string(dst_gray,lang='Hangul+eng'))


cv2.imshow('src',src)
cv2.imshow('dst',dst)
cv2.waitKey()
cv2.destroyAllWindows()