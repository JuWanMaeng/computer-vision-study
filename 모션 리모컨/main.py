import sys
import math
import numpy as np
import cv2


# 카메라 열기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Camera open failed!')
    sys.exit()

w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w2=w//2    #입력영상의 프레임을 반으로 줄일 것임  빠른연산, 노이즈에 덜 민감해짐, 정확도 증가
h2=h//2

ret,frame=cap.read()

frame=cv2.flip(frame,1)    #좌우대칭 - 기본적인 카메라는 반대로 움직이기 때문에 거울처럼 보이기 위함이다
gray1=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
gray1=cv2.resize(gray1,(w2,h2),interpolation=cv2.INTER_AREA)

# 매 프레임에 대해 옵티컬플로우 계산
while True:
    ret,frame=cap.read()

    if not ret:
        print('Frame read failed!')
        break

    frame=cv2.flip(frame,1)  # 좌우대칭
    gray2=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray2=cv2.resize(gray2,(w2,h2),interpolation=cv2.INTER_AREA)

    # 밀집 옵티컬플로우 계산
    flow=cv2.calcOpticalFlowFarneback(gray1,gray2,None,0.5,3,15,3,5,1.1,0) #shpae(h2,w2,2)
    vx,vy=flow[...,0],flow[...,1]
    mag,ang=cv2.cartToPolar(vx,vy)  #직각좌표계-> 극좌표계 벡터의 크기와 각도를 return
    #mag가 움직임이 전혀 없다해서 0이 나온다는것은 보장 못함. 조명성분이 미세함 떨림이 있기 때문이다
    #이러한 부분이 모이면 noise 또는 error 형식으로 동작할수 있다. --> 움직임이 충분히 큰 영역 선택

    # 움직임 벡터 시각화
    hsv=np.zeros((h2,w2,3),dtype=np.uint8)
    hsv[...,0]=ang*180/np.pi/2
    hsv[...,1]=255
    hsv[...,2]=cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.imshow('flow',bgr)

    # 움직임이 충분히 큰 영역 선택
    motion_mask=np.zeros((h2,w2),dtype=np.uint8)
    motion_mask[mag>5.0]=255     #노이즈 움직임을 없애기 위한 과정

    mx=cv2.mean(vx,mask=motion_mask)[0]     # 모션마스크가 흰색인 부분에서 x방향성분에 대한 평균값 4shape 중 [0]이 gray정보
    my=cv2.mean(vy,mask=motion_mask)[0]     #                            y방향성분
    m_mag=math.sqrt(mx*mx+my*my)            #평균 벡터 크기

    if m_mag> 7.0:
        m_ang=math.atan2(my,mx) * 180 / math.pi    # -180~180도 왼쪽이 0도 시계방향으로 90도씩 증가
        m_ang+=180                                 # 0~360도로 바꿔줌

        pt1=(100,100)

        if m_ang>=45 and m_ang<135:
            pt2=(100,30)
        elif m_ang>=135 and m_ang<225:
            pt2=(170,100)
        elif m_ang>225 and m_ang<315:
            pt2=(100,170)
        else:
            pt2=(30,100)

        cv2.arrowedLine(frame,pt1,pt2,(0,0,255),7,cv2.LINE_AA,tipLength=0.7)

    # 결과 영상 화면 출력
    cv2.imshow('frame',frame)
    cv2.imshow('motion_mask',motion_mask)

    if cv2.waitKey(1)==27:
        break

    # 현재 프레임을 이전 프레임으로 복사
    gray1 = gray2

cap.release()
cv2.destroyAllWindows()
