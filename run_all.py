"""
Run FP32 + INT8 on every image in a folder and produce:

  1. annotated_<folder>/*.jpg       -- one side-by-side annotated image per input
                                       (FP32 left, INT8 right). Prefixed MATCH_/DIFF_.
  2. per_image_results_<folder>.csv -- one row per image: detections, top species,
                                       confidence, agreement, and inference time.

Usage:  python3 run_all.py [image_folder]
        (defaults to TBL-1-12-2023--29-01-2024)

Both models are restricted to Leopard / Tiger / Bear (classes 0, 6, 10).
"""

import csv
import glob
import os
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

FOLDER = sys.argv[1] if len(sys.argv) > 1 else "TBL-1-12-2023--29-01-2024"
TAG = os.path.basename(FOLDER.rstrip("/"))
OUTDIR = f"annotated_{TAG}"
CSV = f"per_image_results_{TAG}.csv"
IMGSZ, CONF = 800, 0.6
CLASSES = [0, 6, 10]
NAMES = {0: 'Leopard', 1: 'Cat', 2: 'Dog', 3: 'Deer', 4: 'Goat', 5: 'Monkey',
         6: 'Tiger', 7: 'Wild_boar', 8: 'Cow', 9: 'Hen', 10: 'Bear', 11: 'Byson'}

os.makedirs(OUTDIR, exist_ok=True)
images = sorted(glob.glob(os.path.join(FOLDER, "*.jpg")))
print(f"{len(images)} images")

fp32 = YOLO("train14_best.pt")
int8 = YOLO("train14_best.mlpackage")
# warmup
fp32.predict(images[0], imgsz=IMGSZ, conf=CONF, classes=CLASSES, device="mps", verbose=False)
int8.predict(images[0], imgsz=IMGSZ, conf=CONF, classes=CLASSES, verbose=False)


def summarize(r):
    """(num_detections, top species, top confidence) for a result."""
    b = r.boxes
    if len(b) == 0:
        return 0, "Nothing", 0.0
    confs, clss = b.conf.tolist(), b.cls.tolist()
    i = int(np.argmax(confs))
    return len(b), NAMES[int(clss[i])], float(confs[i])


def banner(im, text, color):
    bar = np.full((44, im.shape[1], 3), color, np.uint8)
    cv2.putText(bar, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return cv2.vconcat([bar, im])


def resize_h(im, h=560):
    return cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))


rows = []
for k, img in enumerate(images, 1):
    t0 = time.perf_counter()
    r32 = fp32.predict(img, imgsz=IMGSZ, conf=CONF, classes=CLASSES,
                       device="mps", verbose=False)[0]
    t_fp32 = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    r8 = int8.predict(img, imgsz=IMGSZ, conf=CONF, classes=CLASSES, verbose=False)[0]
    t_int8 = (time.perf_counter() - t0) * 1000

    n32, sp32, c32 = summarize(r32)
    n8, sp8, c8 = summarize(r8)
    agree = "Yes" if sp32 == sp8 else "No"
    rows.append([os.path.basename(img), n32, sp32, round(c32, 3),
                 n8, sp8, round(c8, 3), agree, round(t_fp32, 1), round(t_int8, 1)])

    # side-by-side annotated image
    l32 = f"FP32  |  {sp32} {c32:.2f}" if n32 else "FP32  |  no detection"
    l8 = f"INT8  |  {sp8} {c8:.2f}" if n8 else "INT8  |  no detection"
    a32 = banner(resize_h(r32.plot()), l32, (90, 90, 90))
    a8 = banner(resize_h(r8.plot()), l8, (150, 90, 30))
    tag = "MATCH" if agree == "Yes" else "DIFF"
    stem = os.path.splitext(os.path.basename(img))[0]
    cv2.imwrite(os.path.join(OUTDIR, f"{tag}_{stem}.jpg"), cv2.hconcat([a32, a8]))

    if k % 40 == 0:
        print(f"  {k}/{len(images)}")

header = ["Image", "FP32_Detections", "FP32_Top_Species", "FP32_Top_Confidence",
          "INT8_Detections", "INT8_Top_Species", "INT8_Top_Confidence",
          "Species_Agreement", "FP32_Time_ms", "INT8_Time_ms"]
with open(CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)

agree_n = sum(1 for r in rows if r[7] == "Yes")
with_det = sum(1 for r in rows if r[2] != "Nothing")
print(f"\nwrote {CSV}  ({len(rows)} rows)")
print(f"wrote {len(rows)} annotated images to {OUTDIR}/")
print(f"images with an animal (FP32): {with_det}/{len(rows)}")
print(f"FP32 vs INT8 species agreement: {agree_n}/{len(rows)} ({agree_n/len(rows)*100:.1f}%)")
