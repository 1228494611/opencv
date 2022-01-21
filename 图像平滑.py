import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


# 添加椒盐噪声
def salt_noise_add(image, prob):
    salt = np.zeros((image.shape), np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.rand()
            if rdn < prob:
                salt[i][j] = 0
            elif rdn > thres:
                salt[i][j] = 255
            else:
                salt[i][j] = image[i][j]
    return salt + image


# 添加高斯噪声
def gaussian_noise_add(image, prob):
    gaussian = np.zeros((image.shape), np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.normal()
            if rdn < prob:
                gaussian[i][j] = 0
            elif rdn > thres:
                gaussian[i][j] = 255
            else:
                gaussian[i][j] = image[i][j]
    return gaussian + image


plt.figure(figsize=(5, 5))
# 添加椒盐噪声
image = cv.imread('pic/test.jpg')
lena = cv.cvtColor(image, cv.COLOR_BGR2RGB)
lenasp = salt_noise_add(lena, 0.4)
plt.subplot(121)
plt.title('noise')
plt.imshow(lenasp)

# 均值滤波
lena1 = cv.blur(lenasp, (3, 3))
plt.subplot(122)
plt.title('filter')
plt.imshow(lena1)
plt.show()

plt.figure()
# 添加高斯噪声
lenagausi = gaussian_noise_add(lena, 0.02)
plt.subplot(121)
plt.title('Gaussian')
plt.imshow(lenagausi)

# 高斯滤波
lena2 = cv.GaussianBlur(lenagausi, (3, 3), 1)
plt.subplot(122)
plt.title('filter')
plt.imshow(lena2)
plt.show()

plt.figure()
plt.subplot(121)
plt.title('noise')
plt.imshow(lenasp)
# 中值滤波(主要处理椒盐噪声)
lena3 = cv.medianBlur(lenasp, 3)
plt.subplot(122)
plt.title('median filter')
plt.imshow(lena3)
plt.show()
