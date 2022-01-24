import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Harris检测
img1 = cv.imread('pic/chess.jpg')
chess = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray = np.float32(chess)
dst = cv.cornerHarris(gray, 2, 3, 0.06)  # 角点检测
img1[dst > 0.001 * dst.max()] = [0, 255, 0]
res1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
plt.title('Harris')
plt.imshow(res1)
plt.show()

# shi—Tomas角点检测
img2 = cv.imread('pic/chess.jpg')
corners = cv.goodFeaturesToTrack(gray, 1000, 0.01, 1)
for i in corners:
    x, y = i.ravel()
    cv.circle(img2, (int(x), int(y)), 2, (0, 255, 0), 1)
res2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
plt.title('shi—Tomas')
plt.imshow(res2)
plt.show()

# sift检测
img3 = cv.imread('pic/chess.jpg')
sift = cv.SIFT_create()  # 创建sift对象
k, dst = sift.detectAndCompute(chess, None)  # 检测关键点并计算关键点描述符
img_sift = cv.drawKeypoints(img3, k, img3, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # 在图像上绘制
res3 = cv.cvtColor(img_sift, cv.COLOR_BGR2RGB)
plt.title('sift')
plt.imshow(res3)
plt.show()

# Fast算法
# 加入非极大值抑制
img4 = cv.imread('pic/chess.jpg')
fast = cv.FastFeatureDetector_create(threshold=30)  # 创建fast对象
kp = fast.detect(img4, None)  # 检测关键点
img_maxSupp = cv.drawKeypoints(img4, kp, None, color=(0, 255, 0))  # 在图像上绘制
res4 = cv.cvtColor(img_maxSupp, cv.COLOR_BGR2RGB)
plt.subplot(121)
plt.title('maxSuppression')
plt.imshow(res4)

# 关闭非极大值抑制
img5 = cv.imread('pic/chess.jpg')
fast.setNonmaxSuppression(0)
kp = fast.detect(img5, None)  # 检测关键点
img_nonmaxSupp = cv.drawKeypoints(img5, kp, None, color=(0, 255, 0))  # 在图像上绘制
res5 = cv.cvtColor(img_nonmaxSupp, cv.COLOR_BGR2RGB)
plt.subplot(122)
plt.title('NonmaxSuppression')
plt.imshow(res5)
plt.show()

# orb检测
img6 = cv.imread('pic/chess.jpg')
orb = cv.ORB_create(nfeatures=1000)
kp, des = orb.detectAndCompute(img6, None)
img_orb = cv.drawKeypoints(img6, kp, img6, None, 0)
res6 = cv.cvtColor(img_orb, cv.COLOR_BGR2RGB)
plt.title('orb')
plt.imshow(res6)
plt.show()
