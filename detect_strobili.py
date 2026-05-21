import cv2
import numpy as np
from pathlib import Path

IMAGE_PATH = "/Users/gvirg/Documents/Cones.png"

# Flowers: vivid pink/red (red wraps in HSV, need two ranges)
FLOWER_LOWER_A = np.array([0,   75, 45])
FLOWER_UPPER_A = np.array([12, 255, 210])
FLOWER_LOWER_B = np.array([168, 75, 45])
FLOWER_UPPER_B = np.array([180, 255, 210])

# Cones: brown/orange — update these by running calibrate.py --cones
CONE_LOWER = np.array([0, 0, 0])
CONE_UPPER = np.array([0, 0, 0])  # zeroed out until calibrated

MIN_AREA = 40
MAX_AREA = 12000


def run():
    img = cv2.imread(IMAGE_PATH)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Build flower mask
    mask_a = cv2.inRange(hsv, FLOWER_LOWER_A, FLOWER_UPPER_A)
    mask_b = cv2.inRange(hsv, FLOWER_LOWER_B, FLOWER_UPPER_B)
    flower_mask = cv2.bitwise_or(mask_a, mask_b)

    # Subtract cone mask if calibrated
    cone_calibrated = not np.all(CONE_LOWER == 0) or not np.all(CONE_UPPER == 0)
    if cone_calibrated:
        cone_mask = cv2.inRange(hsv, CONE_LOWER, CONE_UPPER)
        flower_mask = cv2.bitwise_and(flower_mask, cv2.bitwise_not(cone_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    flower_mask = cv2.morphologyEx(flower_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    flower_mask = cv2.morphologyEx(flower_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(flower_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = img.copy()
    count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_AREA <= area <= MAX_AREA):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        cv2.circle(output, (cx, cy), 8,  (0, 0, 255), -1)
        cv2.circle(output, (cx, cy), 9, (255, 255, 255),  1)
        count += 1
        cv2.putText(output, str(count), (cx + 10, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        cv2.putText(output, str(count), (cx + 10, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)

    cv2.putText(output, f"Flowers detected: {count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    out_path = Path(IMAGE_PATH).with_stem(Path(IMAGE_PATH).stem + "_detected")
    cv2.imwrite(str(out_path), output)
    print(f"Found {count} flowers — saved to {out_path}")


run()
