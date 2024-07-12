import cv2
import numpy as np

img = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
cv2.imshow("img", img)
cv2.waitKey(0)