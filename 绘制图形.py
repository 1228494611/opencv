import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

back = np.zeros((512, 512, 3), np.uint8)  # 生成全黑背景

cv.line(back, (0, 0), (511, 511), (0, 0, 255), 5)  # 图像中添加线段
cv.circle(back, (256, 256), 64, (255, 0, 0), -1)  # 图像中添加圆
cv.rectangle(back, (64, 64), (128, 128), (0, 255, 0), -1)  # 图像中添加矩形
cv.putText(back, 'Fighting', (256, 256), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 5)  # 图像中添加文字

plt.imshow(back)
plt.show()

print(back[100, 100])  # 显示该像素点的三个通道的值
print(back[100, 100, 0])  # 显示该像素点某个通道的值
back[100, 100] = (0, 0, 0)  # 修改该像素点三通道的值

print(back.shape, back.dtype, back.size)  # 打印图像的大小，数据类型，像素个数
