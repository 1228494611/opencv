import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

image = cv.imread('pic/test.jpg')
lena = cv.cvtColor(image, cv.COLOR_BGR2RGB)
plt.imshow(lena)
plt.show()

b, g, r = cv.split(lena)
plt.imshow(b, cmap=plt.cm.gray)
plt.show()

lena2 = cv.merge((b, g, r))
plt.imshow(lena2)
plt.show()

hsv = cv.cvtColor(lena, cv.COLOR_RGB2HSV)
plt.imshow(hsv)
plt.show()

# cv.imwrite('pic/gray.png', lena)
