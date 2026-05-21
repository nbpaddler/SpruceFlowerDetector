"""
Run with --flowers to sample flower colors (pink/red strobili)
Run with --cones   to sample cone colors (brown hanging cones)

Results are printed as lines to paste into detect_strobili.py.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse

IMAGE_PATH = "/Users/gvirg/Documents/Cones.png"

img = cv2.imread(IMAGE_PATH)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
display = img.copy()
samples = []

parser = argparse.ArgumentParser()
parser.add_argument("--cones", action="store_true", help="Sample cone pixels instead of flowers")
args = parser.parse_args()

mode = "cones" if args.cones else "flowers"
dot_color = (0, 140, 255) if mode == "cones" else (0, 255, 255)  # orange for cones, yellow for flowers
title = f"Click on {mode.upper()} — press Q when done"


def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        h, s, v = hsv[y, x]
        samples.append((h, s, v))
        cv2.circle(display, (x, y), 6, dot_color, -1)
        cv2.circle(display, (x, y), 7, (0, 0, 0), 1)
        print(f"Sampled ({x},{y})  HSV=({h},{s},{v})")
        cv2.imshow(title, display)


cv2.namedWindow(title, cv2.WINDOW_NORMAL)
cv2.resizeWindow(title, 1041, 391)
cv2.setMouseCallback(title, on_click)
cv2.imshow(title, display)
print(f"Click on as many {mode} as you can. Press Q when done.")

while True:
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

if not samples:
    print("No samples collected.")
else:
    arr = np.array(samples)
    pad = 25
    h_lo = max(0,   int(arr[:, 0].min()) - pad)
    h_hi = min(180, int(arr[:, 0].max()) + pad)
    s_lo = max(0,   int(arr[:, 1].min()) - pad)
    s_hi = min(255, int(arr[:, 1].max()) + pad)
    v_lo = max(0,   int(arr[:, 2].min()) - pad)
    v_hi = min(255, int(arr[:, 2].max()) + pad)

    print(f"\n--- {len(samples)} {mode} samples ---")
    print(f"H: {arr[:,0].min()}–{arr[:,0].max()}  S: {arr[:,1].min()}–{arr[:,1].max()}  V: {arr[:,2].min()}–{arr[:,2].max()}")

    if mode == "cones":
        print(f"\nPaste into detect_strobili.py:")
        print(f"CONE_LOWER = np.array([{h_lo}, {s_lo}, {v_lo}])")
        print(f"CONE_UPPER = np.array([{h_hi}, {s_hi}, {v_hi}])")
    else:
        print(f"\nPaste into detect_strobili.py:")
        print(f"FLOWER_LOWER_A = np.array([{h_lo}, {s_lo}, {v_lo}])")
        print(f"FLOWER_UPPER_A = np.array([{h_hi}, {s_hi}, {v_hi}])")
