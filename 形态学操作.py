import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

image = cv.imread('pic/erzhi.png')
erzhi = cv.cvtColor(image, cv.COLOR_BGR2RGB)
plt.imshow(erzhi)
plt.show()

plt.figure()
# 图像的膨胀与腐蚀
kernel = np.ones((5, 5), np.uint8)  # 生成卷积核
erode = cv.erode(erzhi, kernel)  # 腐蚀
plt.subplot(121)
plt.imshow(erode)
plt.title('erode')

dilate = cv.dilate(erzhi, kernel)  # 膨胀
plt.subplot(122)
plt.imshow(dilate)
plt.title('dilate')
plt.show()

plt.figure()
# 图像的开闭运算
kernel1 = np.ones((10, 10), np.uint8)
cvopen = cv.morphologyEx(erzhi, cv.MORPH_OPEN, kernel1)  # 图像的开运算(先腐蚀后膨胀)
plt.subplot(121)
plt.imshow(cvopen)
plt.title('open')

cvclose = cv.morphologyEx(erzhi, cv.MORPH_CLOSE, kernel1)  # 图像的闭运算(先膨胀后腐蚀)
plt.subplot(122)
plt.imshow(cvclose)
plt.title('close')
plt.show()

plt.figure()
# 礼帽和黑帽运算
kernel2 = np.ones((5, 5), np.uint8)
top = cv.morphologyEx(erzhi, cv.MORPH_TOPHAT, kernel2)  # 礼帽运算：dst = src - open
plt.subplot(121)
plt.imshow(top)
plt.title('tophat')

black = cv.morphologyEx(erzhi, cv.MORPH_BLACKHAT, kernel2)  # 黑帽运算：dst = close - src
plt.subplot(122)
plt.imshow(black)
plt.title('blackhat')
plt.show()
