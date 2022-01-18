import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

lena = cv.imread('pic/test.jpg')
back = cv.imread('pic/mask(1).jpg')

img1 = cv.cvtColor(lena, cv.COLOR_BGR2RGB)
img2 = cv.cvtColor(back, cv.COLOR_BGR2RGB)

plt.figure(1)
img_cv = cv.add(img1, img2)  # opencv的加法操作为饱和操作
plt.subplot(121)
plt.title('opencv_add')
plt.imshow(img_cv)

img_np = img1 + img2  # np的加法操作为取模操作
plt.subplot(122)
plt.title('np_add')
plt.imshow(img_np)
plt.show()

# 图像的混合
dst = cv.addWeighted(img1, 0.3, img2, 0.7, 0)
plt.figure(2)
plt.imshow(dst)
plt.show()
