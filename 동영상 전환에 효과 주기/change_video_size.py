'''
동영상들의 사이즈는 다들 제각각이므로 사이즈를 맞춰주는 작업이 필요하다
'''

import sys
import cv2
import numpy as np

cap1=cv2.VideoCapture('video1.avi')
cap=cv2.VideoCapture('video2.avi')
fourcc=cv2.VideoWriter_fourcc(*'DIVX')


w=round(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
h=round(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(w,h)

out=cv2.VideoWriter('changed_video2.avi',fourcc,30,(w,h))

while True:
    ret,frame=cap.read()
    if not ret:
        print('fail')
        break

    new_frame=cv2.resize(frame,(w,h),fx=0,fy=0,interpolation=cv2.INTER_CUBIC)
    out.write(new_frame)

cap.release()
out.release()
cv2.destroyAllWindows()

