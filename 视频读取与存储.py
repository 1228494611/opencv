import cv2 as cv
import numpy as py

# 视频读取
video = cv.VideoCapture('pic/video.mkv')

# 判断视频是否读取成功
while video.isOpened():
    # 获取每一帧图像
    ret, frame = video.read()
    if ret:
        cv.imshow('frame',frame)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()
