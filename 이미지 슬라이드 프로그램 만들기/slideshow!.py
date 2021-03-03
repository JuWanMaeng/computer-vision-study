import sys
import glob
import cv2


img_files=glob.glob('.\\images\\*.jpg')    #현재폴더 밑에 images폴더 밑에 ~.jpg파일들을 모두 불러와라
                                           #img_files 리스트에 이미지 파일들을 추가

#사이즈 조절이 가능한 'image'창 생성
cv2.namedWindow('image',cv2.WINDOW_NORMAL)

#전체화면으로 만들수 있는 함수 사용
cv2.setWindowProperty('image',cv2.WND_PROP_FULLSCREEN,
                      cv2.WINDOW_FULLSCREEN)

cnt=len(img_files)
idx=0

while True:
    img=cv2.imread(img_files[idx])

    cv2.imshow('image',img)
    if cv2.waitKey(1000)==27 : #esc
        break
    idx+=1
    if idx>= cnt:
        idx=0

cv2.destroyAllWindows()