import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# 视频获取
video = cv.VideoCapture('pic/facevideo.mkv')

# 判断视频是否读取成功
while True:
    # 获取每一帧图像
    ret, frame = video.read()
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        face_cas = cv.CascadeClassifier("D:/Python/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
        face_cas.load('D:/Python/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
        eye_cas = cv.CascadeClassifier("D:/Python/venv/Lib/site-packages/cv2/data/haarcascade_eye.xml")
        eye_cas.load("D:/Python/venv/Lib/site-packages/cv2/data/haarcascade_eye.xml")
        face_rects = face_cas.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(200, 200))
        for face_rect in face_rects:
            (x, y, w, h) = face_rect
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            roi_color = frame[y:y + h, x:x + w]
            roi_gray = gray[y:y + h, x:x + w]
            eye_rects = eye_cas.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=3, minSize=(45, 45))
            for (x_e, y_e, w_e, h_e) in eye_rects:
                cv.rectangle(roi_color, (x_e, y_e), (x_e + w_e, y_e + h_e), (0, 255, 0), 1)
        cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video.release()  # 释放资源
cv.destroyAllWindows()
