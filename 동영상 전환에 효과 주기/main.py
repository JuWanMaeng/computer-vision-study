import sys
import cv2
import numpy as np

cap1=cv2.VideoCapture('video1.avi')
cap2=cv2.VideoCapture('changed_video2.avi')

if not cap1.isOpened() or not cap2.isOpened():
    print('video open failed!')
    sys.exit()

# 두 동영상의 크기, fps는 같다고 가정함
frame_cnt1=round(cap1.get(cv2.CAP_PROP_FRAME_COUNT)) #비디오 파일의 총 프레임 수
frame_cnt2=round(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
fps=cap1.get(cv2.CAP_PROP_FPS)


#첫번째 영상의 뒷부분 2초, 뒷부분 영상의 앞부분 2초가 겹쳐져서 합성을 할 것임
effect_frames=int(fps*2)
delay=int(1000/fps)

w=round(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
h=round(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc=cv2.VideoWriter_fourcc(*'DIVX')

#출력 동영상 객체 생성
out=cv2.VideoWriter('output.avi',fourcc,fps,(w,h))

for i in range(frame_cnt1-effect_frames):
    ret1,frame1=cap1.read()

    if not ret1:
        break
    
    out.write(frame1)
    cv2.imshow('frame',frame1)
    cv2.waitKey(delay)

for i in range(effect_frames):
    ret1,frame1=cap1.read()
    ret2,frame2=cap2.read()

    #합성
    dx=int(w*i/effect_frames)

    ''' 
    영상 넘기기
    frame=np.zeros((h,w,3),dtype=np.uint8)
    frame[:,0:dx]=frame2[:,0:dx]
    frame[:,dx:w]=frame1[:,dx:w]
    '''
    # 영상 흐려지게 하면서 넘기기
    alpha=1.0-i/effect_frames
    frame=cv2.addWeighted(frame1,alpha,frame2,1-alpha,0)

    out.write(frame)
    cv2.imshow('frame',frame)
    cv2.waitKey(delay)

for i in range(effect_frames,frame_cnt2):
    ret2,frame2=cap2.read()

    if not ret2:
        break
    out.write(frame2)

    cv2.imshow('frame',frame2)
    cv2.waitKey(delay)