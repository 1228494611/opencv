import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

plt.figure()
image = cv.imread('pic/test.jpg', 0)
plt.subplot(121)
plt.imshow(image, plt.cm.gray)

# 图像的灰度直方图
hist = cv.calcHist([image], [0], None, [256], [0, 256])  # 创建灰度直方图
plt.subplot(122)
plt.plot(hist)
plt.show()

# 掩膜图像的灰度直方图(获取图像中的感兴趣区域)
mask = np.zeros(image.shape[:2], np.uint8)
mask[50:150, 200:300] = 1  # 创建蒙版

plt.figure()
mask_image = cv.bitwise_and(image, image, mask=mask)  # 原图像经过掩膜后得到的图像
plt.subplot(121)
plt.imshow(mask_image, cmap=plt.cm.gray)

mask_hist = cv.calcHist([image], [0], mask, [256], [0, 256])  # 掩膜后图像的灰度直方图
plt.subplot(122)
plt.plot(mask_hist)
plt.show()

plt.figure()
# 直方图均衡化
plt.subplot(221)
plt.title('src')
plt.imshow(image, plt.cm.gray)

plt.subplot(222)
plt.title('src')
plt.plot(hist)

dst = cv.equalizeHist(image)  # 直方图均衡化
plt.subplot(223)
plt.title('equalizeHist')
plt.imshow(dst, plt.cm.gray)

dst_hist = cv.calcHist([dst], [0], None, [256], [0, 256])
plt.subplot(224)
plt.title('equalizeHist')
plt.plot(dst_hist)
plt.show()

plt.figure()
# 自适应均衡化
plt.subplot(221)
plt.title('src')
plt.imshow(image, plt.cm.gray)

plt.subplot(222)
plt.title('src')
plt.plot(hist)

cl = cv.createCLAHE(2.0, (8, 8))  # 创建自适应均衡化对象
clahe = cl.apply(image)  # 应用到图像中
plt.subplot(223)
plt.title('clahe')
plt.imshow(clahe, plt.cm.gray)

clahe_hist = cv.calcHist([clahe], [0], None, [256], [0, 256])
plt.subplot(224)
plt.title('clahe')
plt.plot(clahe_hist)
plt.show()
