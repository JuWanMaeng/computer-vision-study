import cv2
import sys
import numpy as np

cap1=cv2.VideoCapture(0)

if not cap1.isOpened():
    print('video open failed!')
    sys.exit()

cap2=cv2.VideoCapture('mountain.mp4')

w=round(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
h=round(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_cnt1=round(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
frame_cnt2=round(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
print('w x h: {} x {}'.format(w,h))
print('frame_cnt1:',frame_cnt1)
print('frame_cnt2:',frame_cnt2)

fps=cap1.get(cv2.CAP_PROP_FPS)
delay=int(1000/fps)

#출력 동영상 객체 생성
fourcc=cv2.VideoWriter_fourcc(*'DIVX')
out=cv2.VideoWriter('output.avi',fourcc,fps,(w,h))

#합성 여부 플래그
do_composit=False    # 합성 안한 상태

while True:
    ret1,frame1=cap1.read()
    if not ret1:
        break

    # do_composit 플래그가 True일 때에만 합성
    if do_composit:
        ret2,frame2=cap2.read()

        if not ret2:
            break
        
        frame2=cv2.resize(frame2,(w,h))

        #HSV 색 공간에서 원하는 영역을 검출하여 합성
        hsv=cv2.cvtColor(frame1,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,(100,150,0),(120,255,200))
        cv2.copyTo(frame2,mask,frame1)
    
    out.write(frame1)
    cv2.imshow('frmae',frame1)
    key=cv2.waitKey(delay)

    #스페이스바를 누르면 do_composit 플래그를 변경
    if key==ord(' '):
        do_composit= not do_composit
    elif key==27:
        break

cap1.release()
cap2.release()
out.release()
cv2.destroyAllWindows()
