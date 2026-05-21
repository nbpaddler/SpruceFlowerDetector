"""
Visualize YOLO-format label files overlaid on their images.
Useful for verifying labels before training.

YOLO label format (one .txt per image, same filename):
    <class_id> <cx> <cy> <w> <h>   (all values normalized 0.0–1.0)

    Example line:  0 0.423 0.318 0.031 0.072
    Means: class 0 (strobilus), centered at 42.3% / 31.8% of image,
           spanning 3.1% width and 7.2% height.

Usage:
    python visualize_labels.py --image path/to/image.png
    python visualize_labels.py --image path/to/image.png --label path/to/label.txt
"""

import cv2
import numpy as np
import argparse
from pathlib import Path

CLASS_NAMES = {0: "strobilus"}
CLASS_COLORS = {0: (0, 220, 80)}


def draw_yolo_labels(image_path, label_path=None):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")

    h, w = img.shape[:2]

    # Auto-find label file if not specified
    if label_path is None:
        label_path = Path(image_path).with_suffix(".txt")

    if not Path(label_path).exists():
        print(f"No label file found at: {label_path}")
        print("Expected format: same name as image, with .txt extension")
        return img, 0

    with open(label_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    count = 0
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])

        # Convert normalized coords to pixel coords
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        color = CLASS_COLORS.get(cls, (255, 255, 0))
        label = CLASS_NAMES.get(cls, str(cls))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        count += 1

    cv2.putText(img, f"Labels: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

    print(f"Rendered {count} labels from {label_path}")
    return img, count


def bootstrap_labels_from_detection(image_path, detections, out_label_path=None):
    """
    Convert detection results from detect_strobili.py into a YOLO label file.
    Use this to pre-label images automatically, then manually correct in a labeling tool.
    """
    img = cv2.imread(str(image_path))
    ih, iw = img.shape[:2]

    if out_label_path is None:
        out_label_path = Path(image_path).with_suffix(".txt")

    lines = []
    for (x, y, bw, bh) in detections:
        cx = (x + bw / 2) / iw
        cy = (y + bh / 2) / ih
        nw = bw / iw
        nh = bh / ih
        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    with open(out_label_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote {len(lines)} labels to {out_label_path}")
    return out_label_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--label", default=None, help="Path to .txt label file")
    parser.add_argument("--save", action="store_true", help="Save annotated image")
    args = parser.parse_args()

    result, count = draw_yolo_labels(args.image, args.label)

    if args.save:
        out = Path(args.image).with_stem(Path(args.image).stem + "_labeled")
        cv2.imwrite(str(out), result)
        print(f"Saved: {out}")

    cv2.imshow("Labels", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
