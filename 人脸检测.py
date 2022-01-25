import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

img = cv.imread('pic/face.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 实例化检测器
face_cas = cv.CascadeClassifier("D:/Python/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
face_cas.load('D:/Python/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

eye_cas = cv.CascadeClassifier("D:/Python/venv/Lib/site-packages/cv2/data/haarcascade_eye.xml")
eye_cas.load("D:/Python/venv/Lib/site-packages/cv2/data/haarcascade_eye.xml")

# 人脸检测
face_rects = face_cas.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
# 绘制人脸检测眼镜
for face_rect in face_rects:
    x, y, w, h = face_rect
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)  # 绘制人脸
    roi_color = img[y:y + h, x:x + w]
    roi_gray = gray[y:y + h, x:x + w]
    eyes_rects = eye_cas.detectMultiScale(roi_gray)
    for (x_e, y_e, w_e, h_e) in eyes_rects:
        cv.rectangle(roi_color, (x_e, y_e), (x_e + w_e, y_e + h_e), (0, 255, 0), 1)  # 绘制眼睛

res = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(res)
plt.show()
