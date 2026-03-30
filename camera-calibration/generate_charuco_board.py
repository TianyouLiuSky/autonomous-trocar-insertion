import cv2
import numpy as np

# Optional: for PDF export with exact physical size
# pip install reportlab pillow
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.units import mm
from PIL import Image  # noqa: F401 (used implicitly by reportlab sometimes)

# Old Params (10cm x 10xm too big for our test setup)
# For a 10x10
# BOARD_MM = 40.0
# SQUARE_MM = 4.0
# MARKER_RATIO = 0.70
# DPI = 600
# MARGIN_MM = 3.0

# For a 13x13
# BOARD_MM = 40.0
# SQUARE_MM = 3.0
# MARKER_RATIO = 0.70
# DPI = 600
# MARGIN_MM = 2.0

# For a 16x16
# BOARD_MM = 40.0
# SQUARE_MM = 2.5
# MARKER_RATIO = 0.65
# DPI = 600
# MARGIN_MM = 2.0

# New Params
# Config A — 24mm board
BOARD_MM = 24.0
SQUARE_MM = 4.0
MARKER_RATIO = 0.70
MARGIN_MM = 2.0

# Config B — 20mm board
# BOARD_MM = 20.0
# SQUARE_MM = 4.0
# MARKER_RATIO = 0.70
# MARGIN_MM = 2.0

# Config C — 21mm board  
# BOARD_MM = 21.0
# SQUARE_MM = 3.5
# MARKER_RATIO = 0.70
# MARGIN_MM = 2.0
DPI = 600

OUT_PNG = "./vision_targets/charuco_24mm.png"
OUT_PDF = "./vision_targets/charuco_24mm.pdf"

# Pick dictionary candidates (auto-selects one large enough)
DICT_CANDIDATES = [
    cv2.aruco.DICT_4X4_50,
    cv2.aruco.DICT_4X4_100,
    cv2.aruco.DICT_4X4_250,
    cv2.aruco.DICT_4X4_1000,
]

# If True, adds a 10 mm scale bar to help verify print scaling
ADD_SCALE_BAR = True
SCALE_BAR_MM = 10.0

# -----------------------------
# Helpers
# -----------------------------
def needed_marker_count(sx, sy) -> int:
    # Approx number of ArUco markers used by OpenCV ChArUco board
    return (sx - 1) * (sy - 1) // 2

def imshow_fit(winname, img, max_w=1200, max_h=800):
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    if s < 1.0:
        img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.imshow(winname, img)

def write_pdf_exact_mm(png_path: str, pdf_path: str, page_w_mm: float, page_h_mm: float):
    from PIL import Image as PILImage
    img = PILImage.open(png_path)
    # Convert to RGB if grayscale (PDF needs RGB)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # DPI tells the PDF renderer the physical size
    img.save(pdf_path, "PDF", resolution=DPI)
    print(f"Saved PDF: {pdf_path}")

# -----------------------------
# Main
# -----------------------------
def main():
    # Grid that fits inside BOARD_MM
    squares_x = int(BOARD_MM // SQUARE_MM)
    squares_y = int(BOARD_MM // SQUARE_MM)

    # mm -> pixels for PNG generation
    px_per_mm = DPI / 25.4
    square_px = int(round(SQUARE_MM * px_per_mm))
    margin_px = int(round(MARGIN_MM * px_per_mm))

    board_w_px = squares_x * square_px
    board_h_px = squares_y * square_px

    # If scale bar, add some extra vertical room inside margin area
    extra_bottom_mm = 8.0 if ADD_SCALE_BAR else 0.0
    extra_bottom_px = int(round(extra_bottom_mm * px_per_mm))

    img_w_px = board_w_px + 2 * margin_px
    img_h_px = board_h_px + 2 * margin_px + extra_bottom_px

    print(f"Grid: {squares_x} x {squares_y} squares")
    print(f"Square: {SQUARE_MM} mm -> {square_px} px at {DPI} DPI")
    print(f"Board active area: {(squares_x*SQUARE_MM):.1f} x {(squares_y*SQUARE_MM):.1f} mm")

    # Choose dictionary large enough
    need = needed_marker_count(squares_x, squares_y)
    chosen_dict = None
    chosen_dict_id = None
    for d in DICT_CANDIDATES:
        dic = cv2.aruco.getPredefinedDictionary(d)
        if dic.bytesList.shape[0] > need:
            chosen_dict = dic
            chosen_dict_id = d
            break
    if chosen_dict is None:
        raise RuntimeError(f"No dictionary large enough for ~{need} markers. Increase dictionary size.")

    print(f"Approx markers needed: {need}")
    print(f"Using dictionary id: {chosen_dict_id} (markers={chosen_dict.bytesList.shape[0]})")

    # Build ChArUco board
    marker_mm = SQUARE_MM * MARKER_RATIO
    board = cv2.aruco.CharucoBoard(
        size=(squares_x, squares_y),
        squareLength=float(SQUARE_MM),
        markerLength=float(marker_mm),
        dictionary=chosen_dict
    )

    board_img = board.generateImage((board_w_px, board_h_px))  # grayscale uint8

    # Canvas (white) with margin and optional bottom area
    canvas = np.full((img_h_px, img_w_px), 255, dtype=np.uint8)
    top = margin_px
    left = margin_px
    canvas[top:top+board_h_px, left:left+board_w_px] = board_img

    # Add scale bar for print verification
    if ADD_SCALE_BAR:
        bar_px = int(round(SCALE_BAR_MM * px_per_mm))
        # Put bar in the bottom area
        y = img_h_px - int(round(4.0 * px_per_mm))
        x0 = margin_px
        x1 = margin_px + bar_px
        cv2.line(canvas, (x0, y), (x1, y), 0, thickness=3)
        cv2.putText(canvas, f"{SCALE_BAR_MM:.0f} mm", (x0, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2, cv2.LINE_AA)

    # Save PNG
    cv2.imwrite(OUT_PNG, canvas)
    print(f"Saved PNG: {OUT_PNG}")

    # Write PDF with exact page size in mm
    # Page includes: active board + margins + extra bottom region (if any)
    page_w_mm = (squares_x * SQUARE_MM) + 2 * MARGIN_MM
    page_h_mm = (squares_y * SQUARE_MM) + 2 * MARGIN_MM + extra_bottom_mm
    write_pdf_exact_mm(OUT_PNG, OUT_PDF, page_w_mm, page_h_mm)
    print(f"Saved PDF: {OUT_PDF}")
    print("Printing tip: print the PDF at 100% / Actual Size (no Fit-to-page). Then measure the scale bar.")

    # Preview
    imshow_fit("ChArUco preview", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()