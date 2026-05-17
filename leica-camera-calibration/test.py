import cv2
import numpy as np
import matplotlib.pyplot as plt

# img = cv2.imread("./single_camera_calibration_data/21Feb2026/24213548_20260221192818.bmp")
img = cv2.imread("./data/21Feb2026/24213548_20260221193127.bmp")

h, w = img.shape[:2]

K = np.array([[8.35021699e+04, 0, 1.22345201e+03],
              [0, 7.64612940e+04, 1.02377758e+03],
              [0, 0, 1]], dtype=np.float64)

dist = np.array([2.30860763e+00, 1.21221044e-04, -8.64131018e-03, 2.12136134e-03, 1.48533434e-08],
                dtype=np.float64)

# Option A: simple undistort (keeps K)
und = cv2.undistort(img, K, dist)

# --- Overlay visualizations ---
alpha = 0.5
blend = cv2.addWeighted(img, alpha, und, 1 - alpha, 0)

diff = cv2.absdiff(img, und)
diff_vis = cv2.convertScaleAbs(diff, alpha=4.0, beta=0)  # boost contrast

plt.figure(figsize=(18,6))

plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB))
plt.title("Blend (0.5 original + 0.5 undistorted)")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
plt.title("Abs diff")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(diff_vis, cv2.COLOR_BGR2RGB))
plt.title("Abs diff (contrast boosted)")
plt.axis("off")

plt.tight_layout()
plt.show()

def canny_edges(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5,5), 0)
    return cv2.Canny(g, 60, 180)

e0 = canny_edges(img)
e1 = canny_edges(und)

overlay = np.zeros((h, w, 3), dtype=np.uint8)
overlay[:, :, 2] = e0  # Red = original edges
overlay[:, :, 1] = e1  # Green = undistorted edges

plt.figure(figsize=(12,6))
plt.imshow(overlay)
plt.title("Edge overlay: Red=original, Green=undistorted (yellow=match)")
plt.axis("off")
plt.show()


plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(und, cv2.COLOR_BGR2RGB))
plt.title("Undistorted")

plt.show()