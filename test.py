import cv2
import numpy as np

img = cv2.imread("./test/imgs/0.jpg", 1)
img2 = cv2.imread("./test/imgs/1.jpg", 1)

img = cv2.resize(img, (100,150))
img2 = cv2.resize(img2, (100,150))
cv2.imshow("img1", img)
cv2.imshow("img2", img2)

cv2.waitKey(0)
cv2.destroyAllWindows()

