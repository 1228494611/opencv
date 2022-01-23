import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

image = cv.imread('pic/test.jpg')
template = cv.imread('pic/template.jpg')
lena = cv.cvtColor(image, cv.COLOR_BGR2RGB)
temp = cv.cvtColor(template, cv.COLOR_BGR2RGB)

# 模板匹配
res = cv.matchTemplate(lena, temp, cv.TM_CCORR)  # 生成模板匹配的矩阵
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)  # 得到模板匹配矩阵中的最大值位置和最小值位置
top_left = max_loc  # 最大值位置为图像的左上角
x, y = temp.shape[:2]
bottom_right = [top_left[0] + x, top_left[1] + y]  # 计算模板的右下角位置
res1 = cv.rectangle(lena, top_left, bottom_right, (0, 255, 0), 2)  # 在原图像中标出
plt.imshow(res1)
plt.show()

# 霍夫线检测
test = cv.imread('pic/huofuxain.jpg')
edge = cv.Canny(test, 50, 150)  # 边缘检测
lines = cv.HoughLines(edge, 0.8, np.pi / 180, 100)  # 获取极坐标下的霍夫线参数

for line in lines:
    rol, theta = line[0]
    a = np.sin(theta)
    b = np.cos(theta)
    x_0 = rol * b
    y_0 = rol * a
    x_1 = int(x_0 + 500 * (-a))
    y_1 = int(y_0 + 500 * b)
    x_2 = int(x_0 - 500 * (-a))
    y_2 = int(y_0 - 500 * b)
    cv.line(test, (x_1, y_1), (x_2, y_2), (0, 0, 255))

res = cv.cvtColor(test, cv.COLOR_BGR2RGB)
plt.imshow(res)
plt.show()

# 霍夫圆检测
img = cv.imread('pic/huofuyuan.jpg')
circle = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# test = cv.Canny(circle,80,40)
# plt.imshow(test,plt.cm.gray)
# plt.show()
circle_filter = cv.medianBlur(circle, 7)
circles = cv.HoughCircles(circle_filter, cv.HOUGH_GRADIENT, 1, 10, param1=80, param2=50, minRadius=0,
                          maxRadius=500)  # 检测霍夫圆
for i in circles[0, :]:
    cv.circle(img, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 2)
    cv.circle(img, (int(i[0]), int(i[1])), 2, (0, 255, 0), 2)

plt.imshow(img)
plt.show()
