import cv2 
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("asset/img.png")
# image = cv2.imread("/-/-/-/untitled folder/asset/img.png")
# print(image.shape)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite("asset/az_grey.jpg" , gray)

cv2.imshow('aziz', gray)
i = cv2.waitKey(0)
if i == 27 or i == ord('s'):
    cv2.destroyAllWindows()

