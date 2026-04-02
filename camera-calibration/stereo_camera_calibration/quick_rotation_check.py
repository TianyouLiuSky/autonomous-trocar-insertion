import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

K_L = np.array([
    [43739.42, 0.0,      1220.32],
    [0.0,      42320.32, 1022.75],
    [0.0,      0.0,      1.0    ]
], dtype=np.float64)
d_L = np.array([16.0047, 0.1633, -0.01154, -0.10767, 0.000712], dtype=np.float64)

K_R = np.array([
    [37599.16, 0.0,      1233.28],
    [0.0,      37162.71, 1024.51],
    [0.0,      0.0,      1.0    ]
], dtype=np.float64)
d_R = np.array([25.1659, -0.2401, 0.10627, 0.06857, 0.001971], dtype=np.float64)

R_cal = np.array([
    [ 0.9748,  0.1221,  0.1868],
    [-0.1198,  0.9925, -0.0235],
    [-0.1883,  0.0005,  0.9821]
], dtype=np.float64)
T_cal = np.array([-156.685, 15.186, -87.961], dtype=np.float64)

img_l = cv2.imread('data/31Mar2026/24213548_20260331121837.bmp', 0)
img_r = cv2.imread('data/31Mar2026/25332589_20260331121837.bmp', 0)

img_l_u = cv2.undistort(img_l, K_L, d_L)
img_r_u = cv2.undistort(img_r, K_R, d_R)

def anaglyph(left, right):
    out = np.zeros((*left.shape, 3), dtype=np.uint8)
    out[:, :, 0] = right  # red   = right
    out[:, :, 1] = left   # cyan  = left (G+B)
    out[:, :, 2] = left
    return out

# Apply R_cal to right image via homography approximation
# H = K_L @ R_cal @ inv(K_R) — projects right onto left's plane
H = K_L @ R_cal @ np.linalg.inv(K_R)
h, w = img_l_u.shape
img_r_warped = cv2.warpPerspective(img_r_u, H, (w, h), flags=cv2.INTER_LINEAR)

overlay_raw    = anaglyph(img_l_u, img_r_u)
overlay_warped = anaglyph(img_l_u, img_r_warped)

rvec, _ = cv2.Rodrigues(R_cal)
deg = np.degrees(np.linalg.norm(rvec))

fig = plt.figure(figsize=(14, 6), facecolor='#0e0e0e')
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.04)

panels = [
    (0, overlay_raw,    f'Raw undistorted overlay  (R ≈ {deg:.1f}° uncorrected)\nL=cyan  R=red'),
    (1, overlay_warped,  'R_cal applied to right image\n(rotation corrected, translation ignored)'),
]

for col, img, title in panels:
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(title, color='#cccccc', fontsize=9, pad=5, fontfamily='monospace')
    ax.axis('off')

fig.text(0.5, 0.01,
         f'stereo_cal_31MAR2026  |  baseline {np.linalg.norm(T_cal):.1f} mm  |  cam_L 24213548  cam_R 25332589',
         ha='center', color='#555555', fontsize=8, fontfamily='monospace')

plt.savefig('stereo_overlay_31MAR2026.png', dpi=150, bbox_inches='tight',
            facecolor='#0e0e0e', edgecolor='none')
plt.show()
print("Saved: stereo_overlay_31MAR2026.png")