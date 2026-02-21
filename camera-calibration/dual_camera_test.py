import cv2
import numpy as np

# ----------------------------
# Config
# ----------------------------
LEFT_PATH  = "./stereo_camera_test_images/Left2.png"
RIGHT_PATH = "./stereo_camera_test_images/Right2.png"

# Display max size (fits most laptop screens)
MAX_W, MAX_H = 1200, 800

# Downsample for alignment speed/stability (ECC runs on these)
ALIGN_SCALE = 0.5

# ECC settings
ECC_MOTION = cv2.MOTION_EUCLIDEAN  # rotation + translation (good first choice)
ECC_ITERS = 200
ECC_EPS = 1e-6
ECC_GAUSS = 5  # smoothing helps on noisy/low-texture images


def imshow_fit(winname, img, max_w=MAX_W, max_h=MAX_H):
    """Show an image resized to fit within max_w x max_h, keeping aspect ratio."""
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.imshow(winname, img)


def load_or_die(path, flags=cv2.IMREAD_COLOR):
    img = cv2.imread(path, flags)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return img


def align_right_to_left_ecc(left_gray, right_gray, scale=ALIGN_SCALE):
    """
    Estimate a Euclidean transform (rotation+translation) that aligns right to left
    using ECC, then return the 2x3 warp matrix in FULL-resolution coordinates.
    """
    # Downsample for ECC
    left_s = cv2.resize(left_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    right_s = cv2.resize(right_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Init warp (2x3 for affine-style warps)
    warp = np.eye(2, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITERS, ECC_EPS)

    # ECC expects float32 images
    left_f = left_s.astype(np.float32) / 255.0
    right_f = right_s.astype(np.float32) / 255.0

    cc, warp = cv2.findTransformECC(
        templateImage=left_f,
        inputImage=right_f,
        warpMatrix=warp,
        motionType=ECC_MOTION,
        criteria=criteria,
        inputMask=None,
        gaussFiltSize=ECC_GAUSS
    )

    # Scale translation back to full-res coordinates
    warp_full = warp.copy()
    warp_full[0, 2] /= scale
    warp_full[1, 2] /= scale

    return cc, warp_full


def main():
    # Load color + grayscale
    left_color = load_or_die(LEFT_PATH, cv2.IMREAD_COLOR)
    right_color = load_or_die(RIGHT_PATH, cv2.IMREAD_COLOR)

    left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

    # Resize right to match left size if needed (ECC requires same dims)
    h, w = left_gray.shape[:2]
    if right_gray.shape[:2] != (h, w):
        right_color = cv2.resize(right_color, (w, h), interpolation=cv2.INTER_AREA)
        right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

    # Align right -> left with ECC
    try:
        cc, warp_full = align_right_to_left_ecc(left_gray, right_gray, scale=ALIGN_SCALE)
        print(f"[ECC] Convergence score (higher is better): {cc:.6f}")
        print("[ECC] Full-res warp (2x3):\n", warp_full)
    except cv2.error as e:
        print("ECC alignment failed. Try changing ALIGN_SCALE, ECC_GAUSS, or use HOMOGRAPHY method.")
        raise e

    # Warp right image into left frame
    aligned_right = cv2.warpAffine(
        right_color, warp_full, (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    )

    # Basic overlay (aligned)
    overlay_aligned = cv2.addWeighted(left_color, 0.5, aligned_right, 0.5, 0)

    # Difference image after alignment (great for "stereo vs duplicate feed" sanity check)
    diff = cv2.absdiff(left_color, aligned_right)

    # Show results
    imshow_fit("Left", left_color)
    imshow_fit("Right (raw)", right_color)
    imshow_fit("Right (aligned to Left)", aligned_right)
    imshow_fit("Overlay (aligned)", overlay_aligned)
    imshow_fit("AbsDiff (aligned)", diff)

    print("\nControls:")
    print(" - Press any key in an image window to close all windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()