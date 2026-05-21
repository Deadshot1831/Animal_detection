"""
Side-by-side FP32 vs INT8 annotated comparison -- BOTH via the CoreML path,
which decodes bounding boxes correctly (the .pt predict() path does not).

  FP32 : train14_fp32.mlpackage   (CoreML, full precision)
  INT8 : train14_best.mlpackage   (CoreML, INT8 quantized)

Because both sides use the correct-box CoreML path, this isolates the *real*
FP32-vs-INT8 difference (quantization only).

Usage:  python3 side_by_side.py [image_folder]

Outputs:
  sidebyside_<folder>/*.jpg   -- FP32 (left) | INT8 (right) per image
  sidebyside_<folder>.csv     -- per-image results
"""

import csv
import glob
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

FOLDER = sys.argv[1] if len(sys.argv) > 1 else "TBL-1-12-2023--29-01-2024"
TAG = os.path.basename(FOLDER.rstrip("/"))
OUTDIR = f"sidebyside_{TAG}"
CSV = f"sidebyside_{TAG}.csv"
IMGSZ, CONF = 800, 0.6
CLASSES = [0, 6, 10]
NAMES = {0: 'Leopard', 1: 'Cat', 2: 'Dog', 3: 'Deer', 4: 'Goat', 5: 'Monkey',
         6: 'Tiger', 7: 'Wild_boar', 8: 'Cow', 9: 'Hen', 10: 'Bear', 11: 'Byson'}

os.makedirs(OUTDIR, exist_ok=True)
images = sorted(glob.glob(os.path.join(FOLDER, "*.jpg")))
print(f"{len(images)} images")

fp32 = YOLO("train14_fp32.mlpackage")  # FP32 CoreML -> correct boxes
int8 = YOLO("train14_best.mlpackage")  # INT8 CoreML -> correct boxes
fp32.predict(images[0], imgsz=IMGSZ, conf=CONF, classes=CLASSES, verbose=False)
int8.predict(images[0], imgsz=IMGSZ, conf=CONF, classes=CLASSES, verbose=False)


def summarize(r):
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
    r32 = fp32.predict(img, imgsz=IMGSZ, conf=CONF, classes=CLASSES, verbose=False)[0]
    r8 = int8.predict(img, imgsz=IMGSZ, conf=CONF, classes=CLASSES, verbose=False)[0]
    n32, sp32, c32 = summarize(r32)
    n8, sp8, c8 = summarize(r8)
    agree = "Yes" if sp32 == sp8 else "No"
    rows.append([os.path.basename(img), n32, sp32, round(c32, 3),
                 n8, sp8, round(c8, 3), agree])

    l32 = f"FP32  |  {sp32} {c32:.2f}" if n32 else "FP32  |  no detection"
    l8 = f"INT8  |  {sp8} {c8:.2f}" if n8 else "INT8  |  no detection"
    a32 = banner(resize_h(r32.plot()), l32, (90, 90, 90))
    a8 = banner(resize_h(r8.plot()), l8, (150, 90, 30))
    tag = "MATCH" if agree == "Yes" else "DIFF"
    stem = os.path.splitext(os.path.basename(img))[0]
    cv2.imwrite(os.path.join(OUTDIR, f"{tag}_{stem}.jpg"), cv2.hconcat([a32, a8]))

    if k % 40 == 0:
        print(f"  {k}/{len(images)}")

with open(CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Image", "FP32_Detections", "FP32_Top_Species", "FP32_Top_Confidence",
                "INT8_Detections", "INT8_Top_Species", "INT8_Top_Confidence",
                "Species_Agreement"])
    w.writerows(rows)

agree_n = sum(1 for r in rows if r[7] == "Yes")
print(f"\nwrote {CSV}  ({len(rows)} rows)")
print(f"wrote {len(rows)} side-by-side images to {OUTDIR}/")
print(f"FP32 vs INT8 species agreement: {agree_n}/{len(rows)} ({agree_n/len(rows)*100:.1f}%)")
