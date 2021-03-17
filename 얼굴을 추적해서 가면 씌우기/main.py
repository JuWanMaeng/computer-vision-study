import sys
import numpy as np
import cv2

# 3채널 img 영상에 4채널 item 영상을 pos 위치에 합성
def overlay(img,mask,pos):

    # 실제 합성을 수행할 부분 영상 좌표 계산
    sx=pos[0]                        
    ex=pos[0] + mask.shape[1]     
    sy=pos[1]
    ey=pos[1] + mask.shape[0]

    # 합성할 영역이 입력 영상 크기를 벗어나면 무시  얼굴을 너무 가까이 가져가면 안됨
    if sy<0 or sy<0 or ex>img.shape[1] or ey>img.shape[0]:
        return


    # 부분 영상 참조. img1: 입력 영상의 부분 영상, img2: 안경 영상의 부분 영상
    img1=img[sy:ey,sx:ex]   # shape=(h,w,3)
    img2=mask  # shape=(h,w,3) mask 3채널(bgr)만 사용
    temp=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)  #gray채널
    alpha= 1. -(temp/255.)   #shape=(h,w)  glasses의 gray채널인 마지막 채널은 weight역할을 할것임
    # 흰부분, 즉 안경사진에서 완전 불투명 부분(합성 되어야 하는 부분)은 알파값이 0이 된다 

    # BGR 채널별로 두 부분 영상의 가중합
    img1[...,0]=(img1[...,0]*alpha + img2[...,0]*(1-alpha)).astype(np.uint8)   #결과값이 실수형 이므로
    img1[...,1]=(img1[...,1]*alpha + img2[...,1]*(1-alpha)).astype(np.uint8)
    img1[...,2]=(img1[...,2]*alpha + img2[...,2]*(1-alpha)).astype(np.uint8)




#카메라 열기
cap=cv2.VideoCapture(0)

if not cap.isOpened():
    print('Failed!')
    sys.exit()

w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc=cv2.VideoWriter_fourcc(*'DIVX')
out=cv2.VideoWriter('output.avi',fourcc,30,(w,h))

# Haar-like XML 파일 열기
face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
eye_classifier=cv2.CascadeClassifier('haarcascade_eye.xml')

if face_classifier.empty() or eye_classifier.empty():
    print('XML load failed!')
    sys.exit()

# 마스크 PNG 파일 열기
mask=cv2.imread('mask.png',cv2.IMREAD_UNCHANGED)

if mask is None:
    print('PNG image open failed!')
    sys.exit()

ew,eh=mask.shape[:2]     # 가로 세로 크기
ex1, ey1=240,300            # 왼쪽 눈 좌표
ex2, ey2=660,300            # 오른쪽 눈 좌표   mask 사진에서 측정한 값들

# 매 프레임에 대해 얼굴 검출 및 안경 합성
while True:
    ret,frame=cap.read()

    if not ret:
        break

    # 얼굴검출
    faces=face_classifier.detectMultiScale(frame,scaleFactor=1.2,minSize=(100,100),maxSize=(400,400))

    for (x,y,w,h) in faces:
        # cv2.rectangle(frame,(x,y,w,h),(255,0,255),2)
        
        # 눈 검출
        faceROI=frame[y:y+h//2,x:x+w]
        eyes=eye_classifier.detectMultiScale(faceROI)

        # 눈을 2개 검출한 것이 아니라면 무시
        if len(eyes) !=2:
            continue

        # 두개의 눈 중앙 위치를 (x1,y1),(x2,y2) 좌표로 저장   
        # eyes는 얼굴검출영상 기준이므로 x가 0인 상황이였다. 그러므로 더해줘야 한다
        x1=x+eyes[0][0] + (eyes[0][2]//2)
        y1=y+eyes[0][1] + (eyes[0][3]//2)
        x2=x+eyes[1][0] + (eyes[1][2]//2)
        y2=y+eyes[1][1] + (eyes[1][3]//2)

        if x1>x2:
            x1,y1,x2,y2=x2,y2,x1,y1

        # 두 눈 사이의 거리를 이용하여 스케일링 팩터를 계산 (두 눈이 수평하다고 가정)
        # 비율을 계산해서 그만큼 resize을 함// 축소이면 INTER_AREA를 사용
        fx=(x2-x1) / (ex2-ex1)
        mask2=cv2.resize(mask,(0,0),fx=fx,fy=fx,interpolation=cv2.INTER_AREA)
        
        # 크기 조절된 안경 영상을 합성할 위치 계산 (좌상단 좌표)
        # 크기 조절된 안경사진이 카메라에 올라가야 하는데 
        # 카메라 눈 좌표에서 안경 눈 좌표를 빼야 안경 사진이 시작하는 좌상단 좌표를 얻을 수 있다
        # 만약 안경 왼쪽 눈 좌표가 (5,5)로 변환되었다 해보자 카메라 왼쪽 눈표가 (20,20)이면
        # 왼쪽 눈과 안경 왼쪽 눈이 딱 맞기 위한 조건은 안경 사진 좌상단이 (15,15)부터 시작해야 맞는다
        pos=(x1-int(ex1*fx),y1-int(ey1*fx))    
        

        # 영상 합성
        overlay(frame,mask2,pos)

    # 프레임 저장 및 화면 출력
    out.write(frame)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()