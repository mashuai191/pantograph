import numpy as np
import cv2

cap = cv2.VideoCapture('20181117_03_08_1750_G1380.MP4')

count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        cv2.imwrite('images/frame{:d}.jpg'.format(count), frame)
        count += 150 # i.e. at 30 fps, this advances one second
        cap.set(1, count)
        
    else:
        cap.release()
        break