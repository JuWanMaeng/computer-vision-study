import cv2
import numpy as np
import sys

def on_mouse(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print(x,y)

src=cv2.imread('mask.png')
print(src.shape)
cv2.namedWindow('src')
cv2.setMouseCallback('src',on_mouse,src)

cv2.imshow('src',src)
cv2.waitKey()

cv2.destroyAllWindows()