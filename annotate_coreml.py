"""
Annotate every image in TBL-1-12-2023--29-01-2024 using the CoreML model.

Fix #1: the .pt predict() path in this ultralytics version corrupts bounding
boxes; the CoreML path produces correct boxes. So annotation is done with the
CoreML model (train14_best.mlpackage).

Outputs:
  annotated_TBL_coreml/*.jpg  -- each input image with correct detection boxes
  results_TBL_coreml.csv      -- per-image detections
"""

import csv
import glob
import os

import cv2
import numpy as np
from ultralytics import YOLO

FOLDER = "TBL-1-12-2023--29-01-2024"
OUTDIR = "annotated_TBL_coreml"
IMGSZ, CONF = 800, 0.6
CLASSES = [0, 6, 10]
NAMES = {0: 'Leopard', 1: 'Cat', 2: 'Dog', 3: 'Deer', 4: 'Goat', 5: 'Monkey',
         6: 'Tiger', 7: 'Wild_boar', 8: 'Cow', 9: 'Hen', 10: 'Bear', 11: 'Byson'}

os.makedirs(OUTDIR, exist_ok=True)
images = sorted(glob.glob(os.path.join(FOLDER, "*.jpg")))
print(f"{len(images)} images")

model = YOLO("train14_best.mlpackage")  # CoreML -> correct boxes
model.predict(images[0], imgsz=IMGSZ, conf=CONF, classes=CLASSES, verbose=False)  # warmup

rows = []
for k, img in enumerate(images, 1):
    r = model.predict(img, imgsz=IMGSZ, conf=CONF, classes=CLASSES, verbose=False)[0]
    cv2.imwrite(os.path.join(OUTDIR, os.path.basename(img)), r.plot())

    b = r.boxes
    cls, conf = b.cls.tolist(), b.conf.tolist()
    all_det = "; ".join(f"{NAMES[int(c)]}:{cf:.2f}" for c, cf in zip(cls, conf)) or "Nothing"
    if cls:
        i = int(np.argmax(conf))
        top_sp, top_cf = NAMES[int(cls[i])], round(float(conf[i]), 3)
    else:
        top_sp, top_cf = "Nothing", 0.0
    rows.append([os.path.basename(img), len(cls), top_sp, top_cf, all_det])

    if k % 40 == 0:
        print(f"  {k}/{len(images)}")

with open("results_TBL_coreml.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Image", "Num_Detections", "Top_Species", "Top_Confidence", "All_Detections"])
    w.writerows(rows)

with_animal = sum(1 for r in rows if r[2] != "Nothing")
print(f"\nwrote results_TBL_coreml.csv ({len(rows)} rows)")
print(f"wrote {len(rows)} annotated images to {OUTDIR}/")
print(f"images with an animal: {with_animal}/{len(rows)}")
