import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

image = cv.imread('pic/test.jpg', 0)
plt.imshow(image, plt.cm.gray)
plt.show()

plt.figure(2)
# sobel算子
x = cv.Sobel(image, cv.CV_16S, 1, 0)  # x方向的梯度
y = cv.Sobel(image, cv.CV_16S, 0, 1)  # y方向的梯度
# 截断处理
absx = cv.convertScaleAbs(x)
absy = cv.convertScaleAbs(y)
result = cv.addWeighted(absx, 0.5, absy, 0.5, 0)  # x，y图像的融合
plt.subplot(121)
plt.title('sobel')
plt.imshow(result, plt.cm.gray)

# scharr算子
x = cv.Sobel(image, cv.CV_16S, 1, 0, ksize=-1)  # x方向的梯度
y = cv.Sobel(image, cv.CV_16S, 0, 1, ksize=-1)  # y方向的梯度
# 截断处理
absx = cv.convertScaleAbs(x)
absy = cv.convertScaleAbs(y)

result1 = cv.addWeighted(absx, 0.5, absy, 0.5, 0)  # x，y图像的融合
plt.subplot(122)
plt.title('scharr')
plt.imshow(result1, plt.cm.gray)
plt.show()
