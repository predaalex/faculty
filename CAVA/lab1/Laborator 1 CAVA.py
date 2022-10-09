# 1.2
import os

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# print('versiunea:', cv.__version__)

# # 1.3 + 1.5
# img = cv.imread("butterfly.jpeg")
# cv.imshow("Fluture galben",img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# # 1.4
img = cv.imread("football.jpg",cv.IMREAD_GRAYSCALE)
# cv.imshow("Fluture gray",img)
# cv.waitKey(0)
# cv.destroyAllWindows()
#
# H, W = img.shape
# print(H,W)
#
# # 1.6
img = cv.resize(img,(100, 100))
H, W = img.shape
# print(H,W)
cv.imshow("Fluture gray redimensionat",img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# a
v = img.flatten()
x = np.sort(v)
print(x)
# plt.plqot(np.arange(len(x)), x)
# plt.show()

# b
A = img[50:,50:].copy()

# cv.imshow('test', A)
# cv.waitKey(0)
# cv.destroyAllWindows()


# c
t = np.median(x)
print(t)

# d

B = img.copy()
B[B < t] = 0
B[B >= t] = 255

# cv.imshow('test', B)
# cv.waitKey(0)
# cv.destroyAllWindows()


# e
eMediu = img.mean()
C = img - eMediu
C[ C < 0 ] = 0
C = np.uint8(C)


# f
iMin = img.min()
print(iMin)
l, c = np.where(img == iMin)
print(l, c)

# 1.7
# a + b
dir = 'colectiiImagini/set1/'

files = os.listdir(dir)
print(files)

colorImg = []
grayImg = []

for file in files:
    patch = dir + file
    img = cv.imread(patch)
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    grayImg.append(imgGray)
    colorImg.append(img)

colorImg = np.array(colorImg)
grayImg = np.array(grayImg)

meanColorImg = np.uint8(np.mean(colorImg, axis = 0))
meanGreyImg = np.uint8(np.mean(grayImg, axis = 0))
print(meanColorImg)

# deviatia standart
# C

X = np.uint8(np.std(grayImg, axis = 0))

# 1.8

img = cv.imread('butterfly.jpeg')
dim = 20
# H1, W1 = imagine.shape

y = np.random.randint(0, 670 - dim, size = 500)
x = np.random.randint(0, 626 - dim, size = 500)

crop = img[250:250+dim,250:250+dim, :].copy()

dist = np.zeros(0)
for i in range(0, 500):
    patch = img[y[i]:y[i]+dim, x[i]:x[i]+dim, :].copy()
    dist[i] = np.sqrt(np.sum((patch - crop))**2)
print(dist)
index = np.argmin(dist)
print(index)

imgNew = img.copy()


