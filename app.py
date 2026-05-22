import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io

st.cache_data.clear()

# HSV ranges for flowers (red wraps in HSV, need two ranges)
FLOWER_LOWER_A = np.array([0,   75, 45])
FLOWER_UPPER_A = np.array([12, 255, 210])
FLOWER_LOWER_B = np.array([168, 75, 45])
FLOWER_UPPER_B = np.array([180, 255, 210])

MIN_AREA_FRAC = 0.00010
MAX_AREA_FRAC = 0.02950

GRID_COLOR  = (255, 215, 0)   # cyan in BGR
GRID_LABELS = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]


def blob_confidence(cnt):
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0.0
    return cv2.contourArea(cnt) / hull_area


def detect_flowers(img_array, scale, show_grid):
    h_img, w_img = img_array.shape[:2]
    img_area = h_img * w_img

    min_area = max(10, int(img_area * MIN_AREA_FRAC * (scale ** 2)))
    max_area = int(img_area * MAX_AREA_FRAC * (scale ** 2))

    hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
    mask_a = cv2.inRange(hsv, FLOWER_LOWER_A, FLOWER_UPPER_A)
    mask_b = cv2.inRange(hsv, FLOWER_LOWER_B, FLOWER_UPPER_B)
    mask = cv2.bitwise_or(mask_a, mask_b)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = img_array.copy()
    count = 0
    solidities = []
    diameters = []
    centers = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area):
            continue

        score = blob_confidence(cnt)
        solidities.append(score)

        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        diameters.append(min(w, h))
        centers.append((cx, cy))

        dot_r = max(3, min(w, h) // 3)

        if score >= 0.75:
            dot_color = (0, 200, 0)
        elif score >= 0.55:
            dot_color = (0, 200, 255)
        else:
            dot_color = (0, 0, 255)

        cv2.circle(output, (cx, cy), dot_r,     dot_color,      -1)
        cv2.circle(output, (cx, cy), dot_r + 1, (255, 255, 255), 1)

        count += 1
        cv2.putText(output, str(count), (cx + dot_r + 2, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        cv2.putText(output, str(count), (cx + dot_r + 2, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # ── Quadrant grid ────────────────────────────────────────────────────────
    mid_x = w_img // 2
    mid_y = h_img // 2

    quadrant_counts = [0, 0, 0, 0]  # TL, TR, BL, BR
    for (cx, cy) in centers:
        col = 0 if cx < mid_x else 1
        row = 0 if cy < mid_y else 1
        quadrant_counts[row * 2 + col] += 1

    if show_grid:
        thickness = max(2, w_img // 400)
        font_scale_grid = max(0.8, w_img / 1000)

        # Grid lines
        cv2.line(output, (mid_x, 0),     (mid_x, h_img), GRID_COLOR, thickness)
        cv2.line(output, (0, mid_y),     (w_img, mid_y), GRID_COLOR, thickness)
        cv2.rectangle(output, (0, 0), (w_img - 1, h_img - 1), GRID_COLOR, thickness)

        # Per-quadrant count labels
        quadrant_centers = [
            (mid_x // 2,           mid_y // 2),
            (mid_x + mid_x // 2,   mid_y // 2),
            (mid_x // 2,           mid_y + mid_y // 2),
            (mid_x + mid_x // 2,   mid_y + mid_y // 2),
        ]
        for (lx, ly), qc, label in zip(quadrant_centers, quadrant_counts, GRID_LABELS):
            text = f"{label}: {qc}"
            cv2.putText(output, text, (lx - 60, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_grid, (255, 255, 255), 4)
            cv2.putText(output, text, (lx - 60, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_grid, GRID_COLOR, 2)

    # ── Summary banner ───────────────────────────────────────────────────────
    avg_confidence = round(float(np.mean(solidities)) * 100, 1) if solidities else 0.0
    avg_diameter   = round(float(np.mean(diameters)),  1)        if diameters  else 0.0

    if avg_diameter == 0:
        scale_description = "N/A"
    elif avg_diameter < 8:
        scale_description = f"{avg_diameter}px — Very far away"
    elif avg_diameter < 15:
        scale_description = f"{avg_diameter}px — Far"
    elif avg_diameter < 25:
        scale_description = f"{avg_diameter}px — Medium distance"
    elif avg_diameter < 40:
        scale_description = f"{avg_diameter}px — Close"
    else:
        scale_description = f"{avg_diameter}px — Very close / macro"

    if count == 0:
        confidence_label = "N/A"
        notes = "No flowers detected — try adjusting the scale slider"
    elif avg_confidence >= 75:
        confidence_label = f"{avg_confidence}% — High"
        notes = "Detections look clean"
    elif avg_confidence >= 55:
        confidence_label = f"{avg_confidence}% — Medium"
        notes = "Some detections may be noise or cones — review image"
    else:
        confidence_label = f"{avg_confidence}% — Low"
        notes = "Many irregular blobs — check scale or lighting"

    font_scale = max(0.6, w_img / 1200)
    cv2.putText(output, f"Flowers: {count}  |  Conf: {avg_confidence}%", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 3)
    cv2.putText(output, f"Flowers: {count}  |  Conf: {avg_confidence}%", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 180), 2)

    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return count, confidence_label, avg_confidence, notes, avg_diameter, scale_description, quadrant_counts, output_rgb


def pil_to_bgr(pil_img):
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def img_to_download_bytes(img_rgb):
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")


# ── UI ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Spruce Flower Detector", layout="wide")
st.title("Spruce Flower Detector")
st.caption("Upload one or more photos of white spruce trees to count flowers.")

if "results" not in st.session_state:
    st.session_state.results = {}

with st.sidebar:
    st.header("Settings")

    scale = st.select_slider(
        "Camera distance / flower size",
        options=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0],
        value=1.0,
        format_func=lambda v: {
            0.25: "Very far",
            0.5:  "Far",
            0.75: "Slightly far",
            1.0:  "Reference (default)",
            1.5:  "Slightly close",
            2.0:  "Close",
            3.0:  "Very close",
            4.0:  "Extreme close-up",
        }[v]
    )
    st.caption("Increase if flowers appear large (close-up). Decrease if they appear small (far away).")

    st.divider()
    show_grid = st.toggle("Show quadrant grid", value=True)
    st.caption("Divides each image into 4 quadrants and counts flowers per section.")

    st.divider()
    with st.expander("How confidence works"):
        st.markdown("""
Each blob is scored by **solidity** (filled area ÷ convex hull area).
Clean strobili are compact; noise and partial cones are irregular.

Dot colours:
- **Green** ≥ 75%
- **Yellow** 55–74%
- **Red** < 55%
        """)

uploaded_files = st.file_uploader(
    "Choose images",
    type=["png", "jpg", "jpeg", "tiff"],
    accept_multiple_files=True
)

if uploaded_files:
    cache_key = lambda name: (name, scale, show_grid)
    new_files = [f for f in uploaded_files if cache_key(f.name) not in st.session_state.results]

    if new_files:
        with st.spinner(f"Processing {len(new_files)} image(s)..."):
            for uploaded_file in new_files:
                pil_img = Image.open(uploaded_file)
                bgr = pil_to_bgr(pil_img)
                count, confidence_label, conf_raw, notes, avg_diameter, scale_desc, quadrant_counts, annotated = detect_flowers(bgr, scale, show_grid)
                st.session_state.results[cache_key(uploaded_file.name)] = {
                    "filename":         uploaded_file.name,
                    "count":            count,
                    "confidence_label": confidence_label,
                    "conf_raw":         conf_raw,
                    "notes":            notes,
                    "avg_diameter":     avg_diameter,
                    "scale_desc":       scale_desc,
                    "quadrant_counts":  quadrant_counts,
                    "annotated":        annotated,
                    "download_bytes":   img_to_download_bytes(annotated),
                    "scale":            scale,
                }

    uploaded_names = {f.name for f in uploaded_files}
    for key in list(st.session_state.results.keys()):
        if key[0] not in uploaded_names:
            del st.session_state.results[key]

results = [v for k, v in st.session_state.results.items() if k[1] == scale and k[2] == show_grid]

if results:
    st.subheader("Summary")
    df = pd.DataFrame([{
        "Image":            r["filename"],
        "Total Flowers":    r["count"],
        "Top-Left":         r["quadrant_counts"][0],
        "Top-Right":        r["quadrant_counts"][1],
        "Bottom-Left":      r["quadrant_counts"][2],
        "Bottom-Right":     r["quadrant_counts"][3],
        "Avg Flower Size":  r["scale_desc"],
        "Confidence":       r["confidence_label"],
        "Notes":            r["notes"],
    } for r in results])

    st.dataframe(df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    col1.metric("Total flowers across all images", df["Total Flowers"].sum())
    col2.metric("Images processed", len(results))

    csv_df = pd.DataFrame([{
        "Image":                    r["filename"],
        "Total Flowers":            r["count"],
        "Quadrant: Top-Left":       r["quadrant_counts"][0],
        "Quadrant: Top-Right":      r["quadrant_counts"][1],
        "Quadrant: Bottom-Left":    r["quadrant_counts"][2],
        "Quadrant: Bottom-Right":   r["quadrant_counts"][3],
        "Avg Flower Diameter (px)": r["avg_diameter"],
        "Camera Distance":          r["scale_desc"],
        "Scale Setting":            r["scale"],
        "Confidence (%)":           r["conf_raw"],
        "Confidence Band":          r["confidence_label"],
        "Notes":                    r["notes"],
    } for r in results])

    st.download_button(
        label="Download CSV report",
        data=df_to_csv(csv_df),
        file_name="flower_detection_report.csv",
        mime="text/csv"
    )

    st.divider()
    st.subheader("Labelled Images")

    for r in results:
        with st.expander(f"{r['filename']} — {r['count']} flowers  |  {r['confidence_label']}", expanded=True):
            st.image(r["annotated"], use_container_width=True)

            col_dl, col_del = st.columns([3, 1])
            col_dl.download_button(
                label="Download labelled image",
                data=r["download_bytes"],
                file_name=r["filename"].rsplit(".", 1)[0] + "_detected.png",
                mime="image/png",
                key=f"dl_{r['filename']}_{scale}_{show_grid}"
            )
            if col_del.button("Remove image", key=f"del_{r['filename']}_{scale}_{show_grid}", type="secondary"):
                for key in [k for k in st.session_state.results if k[0] == r["filename"]]:
                    del st.session_state.results[key]
                st.rerun()
