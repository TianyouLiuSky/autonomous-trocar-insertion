import cv2
import numpy as np

# Load images
img1 = cv2.imread("cam1.png")
img2 = cv2.imread("cam2.png")

# Resize if needed (make same size)
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Alpha blend
overlay = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

cv2.imshow("Overlay", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
