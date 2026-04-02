#!/usr/bin/env python3
"""
generate_composite_board.py
Generates the ATI "eye-eye" composite calibration target:
  - Center: 6x6 ChArUco (24x24mm, 4mm squares, 0.70 marker ratio)
  - Corners: 4x AprilTag36h11 (24mm face, IDs 0-3)
  - Edge centers: 4x AprilTag36h11 (24mm face, IDs 4-7)  [optional, set EDGE_TAGS=True]
  - Grid: 3x3 cells, each 28mm (tag/board + 2mm margin per side)
  - Total board: 84x84mm
Output: composite_board_<DATE>.png + .pdf (600 DPI)
"""

import cv2
import cv2.aruco as aruco
import numpy as np
from PIL import Image
import os
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
BOARD_MM     = 24.0      # ChArUco board face (6x6 @ 4mm = 24mm)
SQUARE_MM    = 4.0
MARKER_RATIO = 0.70
MARGIN_MM    = 2.0       # margin around each element → cell = BOARD_MM + 2*MARGIN_MM = 28mm

TAG_MM       = 24.0      # AprilTag face size
EDGE_TAGS    = True      # set False to leave edge cells blank (4 vs 8 tags)

DPI          = 600
OUTPUT_DIR   = "./composite_board_output/"

# ── Derived ───────────────────────────────────────────────────────────────────
CELL_MM      = BOARD_MM + 2 * MARGIN_MM          # 28mm
TOTAL_MM     = 3 * CELL_MM                       # 84mm
PX_PER_MM   = DPI / 25.4                         # 23.62 px/mm at 600 DPI
CELL_PX      = round(CELL_MM * PX_PER_MM)        # pixels per cell
BOARD_PX     = round(BOARD_MM * PX_PER_MM)       # ChArUco board in px
TAG_PX       = round(TAG_MM * PX_PER_MM)         # AprilTag face in px
TOTAL_PX     = 3 * CELL_PX

MARGIN_PX    = (CELL_PX - TAG_PX) // 2          # centering margin
CHARUCO_MARGIN_PX = (CELL_PX - BOARD_PX) // 2

# Dictionaries
charuco_dict  = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
apriltag_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)

# Cell (row, col) → 0-indexed, center cell = (1,1)
# Corner cells: (0,0),(0,2),(2,0),(2,2) → tag IDs 0,1,2,3
# Edge cells:   (0,1),(1,0),(1,2),(2,1) → tag IDs 4,5,6,7
CORNER_CELLS = [(0, 0, 0), (0, 2, 1), (2, 0, 2), (2, 2, 3)]
EDGE_CELLS   = [(0, 1, 4), (1, 0, 5), (1, 2, 6), (2, 1, 7)]

def make_apriltag(tag_id: int, size_px: int) -> np.ndarray:
    """Render a single AprilTag36h11 marker (white border included in size_px)."""
    img = np.ones((size_px, size_px), dtype=np.uint8) * 255
    aruco.generateImageMarker(apriltag_dict, tag_id, size_px, img, 1)
    return img

def make_charuco_board() -> np.ndarray:
    """Render the 6x6 ChArUco board at BOARD_PX x BOARD_PX."""
    board = aruco.CharucoBoard(
        (6, 6),
        SQUARE_MM / 1000.0,
        (SQUARE_MM * MARKER_RATIO) / 1000.0,
        charuco_dict
    )
    img = board.generateImage((BOARD_PX, BOARD_PX), marginSize=0, borderBits=1)
    return img

def place_cell(canvas: np.ndarray, row: int, col: int,
               patch: np.ndarray, offset_px: int) -> None:
    """Paste patch into (row, col) cell with centering offset."""
    r0 = row * CELL_PX + offset_px
    c0 = col * CELL_PX + offset_px
    h, w = patch.shape
    canvas[r0:r0 + h, c0:c0 + w] = patch

def main():
    canvas = np.ones((TOTAL_PX, TOTAL_PX), dtype=np.uint8) * 255

    # Center ChArUco
    charuco_img = make_charuco_board()
    place_cell(canvas, 1, 1, charuco_img, CHARUCO_MARGIN_PX)

    # Corner AprilTags
    for row, col, tag_id in CORNER_CELLS:
        tag_img = make_apriltag(tag_id, TAG_PX)
        place_cell(canvas, row, col, tag_img, MARGIN_PX)

    # Edge AprilTags (optional)
    if EDGE_TAGS:
        for row, col, tag_id in EDGE_CELLS:
            tag_img = make_apriltag(tag_id, TAG_PX)
            place_cell(canvas, row, col, tag_img, MARGIN_PX)

    # Add thin registration crosshairs at cell intersections (optional, aids manual inspection)
    for i in [CELL_PX, 2 * CELL_PX]:
        canvas[i - 1:i + 1, :] = 180      # horizontal
        canvas[:, i - 1:i + 1] = 180      # vertical

    # Save PNG
    date_str  = datetime.now().strftime('%d%b%Y').upper()
    n_tags    = 8 if EDGE_TAGS else 4
    base_name = f"composite_board_{n_tags}tags_{date_str}"

    png_path = os.path.join(OUTPUT_DIR, base_name + ".png")
    cv2.imwrite(png_path, canvas)
    print(f"PNG saved: {png_path}  ({TOTAL_PX}x{TOTAL_PX}px @ {DPI}DPI)")

    # Save PDF via Pillow (600 DPI, no downsampling)
    pil_img  = Image.fromarray(canvas).convert("RGB")
    pdf_path = os.path.join(OUTPUT_DIR, base_name + ".pdf")
    pil_img.save(pdf_path, "PDF", resolution=DPI)
    print(f"PDF saved: {pdf_path}")

    # Print board summary
    print(f"\n── Board summary ──────────────────────────────")
    print(f"  Cell size:         {CELL_MM:.1f}mm × {CELL_MM:.1f}mm  ({CELL_PX}px)")
    print(f"  ChArUco board:     {BOARD_MM:.1f}mm × {BOARD_MM:.1f}mm  ({BOARD_PX}px)  — 6×6 squares @ {SQUARE_MM}mm")
    print(f"  AprilTag face:     {TAG_MM:.1f}mm × {TAG_MM:.1f}mm  ({TAG_PX}px)  — IDs 0-{n_tags-1}, DICT_APRILTAG_36h11")
    print(f"  Total board:       {TOTAL_MM:.1f}mm × {TOTAL_MM:.1f}mm  ({TOTAL_PX}px)")
    print(f"  DPI:               {DPI}")
    print(f"  Marker ratio:      {MARKER_RATIO}")
    print(f"  Edge tags:         {'yes (IDs 4–7)' if EDGE_TAGS else 'no'}")
    print(f"\n  → Print at exactly {DPI} DPI. Verify BOARD_MM and TAG_MM with calipers before use.")

if __name__ == "__main__":
    main()