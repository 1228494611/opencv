import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# 获取视频
video = cv.VideoCapture('pic/video3.mkv')
ret, frame = video.read()  # 获取图像的第一帧
res = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
# r, h, c, w = 240, 100, 265, 380
# test = cv.rectangle(frame, (r, c), (r + h, c + w), (255, 0, 0), 1)
# plt.imshow(test)
# plt.show()

# 指定追踪目标
r, h, c, w = 100, 500, 255, 50
win = (c, r, w, h)
roi = frame[r:r + h, c:c + w]

# 计算目标直方图
roi_hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)  # 转换为HSV通道
roi_hist = cv.calcHist([roi_hsv], [0], None, [180], [0, 100])  # 计算目标直方图
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)  # 直方图归一化

# 目标追踪
term = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
while True:
    ret, frame = video.read()
    if ret:
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([frame_hsv], [0], roi_hist, [0, 180], 1)
        ret, win = cv.meanShift(dst, win, term)
        # ret, win = cv.CamShift(dst, win, term)

        x, y, w, h = win
        # pts = cv.boxPoints(ret)
        # pts = np.int0(pts)
        # img = cv.polylines(frame, [pts], True, (0, 0, 255), 1)
        img = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv.imshow("frame", img)
        if cv.waitKey(25) & 0xFF == ord('q'):
            break

# 释放资源
video.release()
cv.destroyAllWindows()
