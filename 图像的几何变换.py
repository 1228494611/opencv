import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

image = cv.imread('pic/test.jpg')

rows, cols = image.shape[:2]
print(rows, cols)
print(image.shape)

plt.figure(1)

# 图像的绝对尺寸缩放
image_exp = cv.resize(image, dsize=(cols * 2, rows * 2))
lena1 = cv.cvtColor(image_exp, cv.COLOR_BGR2RGB)
print(lena1.shape)
plt.subplot(121)
plt.imshow(lena1)

# 图像相对尺寸缩放
image_exp1 = cv.resize(image, dsize=None, fx=0.5, fy=0.5)
lena2 = cv.cvtColor(image_exp1, cv.COLOR_BGR2RGB)
print(lena2.shape)
plt.subplot(122)
plt.imshow(lena2)
plt.show()

# 图像的平移
M = np.float32([[1, 0, 100], [0, 1, 50]])
image_tran = cv.warpAffine(image, M=M, dsize=(cols, rows))
lena3 = cv.cvtColor(image_tran, cv.COLOR_BGR2RGB)
plt.imshow(lena3)
plt.show()

# 图像的旋转
M = cv.getRotationMatrix2D((cols / 2, rows / 2), 45, 0.5)  # 获取旋转矩阵
image_rota = cv.warpAffine(image, M=M, dsize=(cols, rows))
lena4 = cv.cvtColor(image_rota, cv.COLOR_BGR2RGB)
plt.imshow(lena4)
plt.show()

# 图像的仿射变换
pta1 = np.float32([[50, 50], [100, 150], [200, 250]])
pta2 = np.float32([[100, 100], [200, 200], [210, 240]])
M = cv.getAffineTransform(pta1, pta2)  # 获取仿射变换矩阵，需要三个点
image_fs = cv.warpAffine(image, M, dsize=(cols, rows))
lena5 = cv.cvtColor(image_fs, cv.COLOR_BGR2RGB)
plt.imshow(lena5)
plt.show()

# 图像的投射变换
pta3 = np.float32([[50, 50], [100, 150], [200, 250], [120, 120]])
pta4 = np.float32([[100, 100], [200, 200], [210, 240], [224, 300]])
M1 = cv.getPerspectiveTransform(pta3, pta4)
image_ts = cv.warpPerspective(image, M1, dsize=(cols, rows))
lena6 = cv.cvtColor(image_ts, cv.COLOR_BGR2RGB)
plt.imshow(lena6)
plt.show()

plt.figure(2)
# 图像金字塔

# 图像上采样
imgup0 = cv.cvtColor(cv.pyrUp(image), cv.COLOR_BGR2RGB)
imgup1 = cv.pyrUp(imgup0)
imgup2 = cv.pyrUp(imgup1)
plt.subplot(131)
plt.imshow(imgup0)
plt.subplot(132)
plt.imshow(imgup1)
plt.subplot(133)
plt.imshow(imgup2)
plt.show()

plt.figure(3)
# 图像下采样
imgdown0 = cv.cvtColor(cv.pyrUp(image), cv.COLOR_BGR2RGB)
imgdown1 = cv.pyrDown(imgdown0)
imgdown2 = cv.pyrDown(imgdown1)
plt.subplot(131)
plt.imshow(imgdown0)
plt.subplot(132)
plt.imshow(imgdown1)
plt.subplot(133)
plt.imshow(imgdown2)
plt.show()
