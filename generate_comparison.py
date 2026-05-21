"""
Generate the FP32-vs-INT8 comparison artifacts on TBL-1-12-2023--29-01-2024:

  1. comparison_metrics.csv   -- speed + accuracy metrics for FP32 / FP16 / INT8
  2. comparison_output/*.jpg  -- side-by-side annotated images (FP32 | INT8)

The folder has no ground-truth labels, so the FP32 PyTorch model is used as the
accuracy reference; FP16 and INT8 are scored on agreement with it.
"""

import csv
import glob
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

FOLDER = "TBL-1-12-2023--29-01-2024"
OUTDIR = "comparison_output"
IMGSZ, CONF = 800, 0.6
CLASSES = [0, 6, 10]  # detect only Leopard, Tiger, Bear
NAMES = {0: 'Leopard', 1: 'Cat', 2: 'Dog', 3: 'Deer', 4: 'Goat', 5: 'Monkey',
         6: 'Tiger', 7: 'Wild_boar', 8: 'Cow', 9: 'Hen', 10: 'Bear', 11: 'Byson'}
DANGER = {0, 6, 10}  # Leopard, Tiger, Bear

os.makedirs(OUTDIR, exist_ok=True)
images = sorted(glob.glob(os.path.join(FOLDER, "*.jpg")))
n = len(images)
print(f"{n} images\n")

fp32 = YOLO("train14_best.pt")
int8 = YOLO("train14_best.mlpackage")


# --------------------------------------------------------------------------
# 1. Run every model over the folder
# --------------------------------------------------------------------------
def run_all(model, **kw):
    model.predict(images[0], imgsz=IMGSZ, conf=CONF, classes=CLASSES,
                  verbose=False, **kw)  # warmup
    dets, t0 = [], time.perf_counter()
    for img in images:
        b = model.predict(img, imgsz=IMGSZ, conf=CONF, classes=CLASSES,
                          verbose=False, **kw)[0].boxes
        dets.append({"cls": [int(c) for c in b.cls.tolist()],
                     "conf": b.conf.tolist(), "xyxy": b.xyxy.tolist()})
    return dets, time.perf_counter() - t0


print("Running FP32 ..."); fp32_d, fp32_t = run_all(fp32, device="mps")
print("Running FP16 ..."); fp16_d, fp16_t = run_all(fp32, device="mps", half=True)
print("Running INT8 ..."); int8_d, int8_t = run_all(int8)


# --------------------------------------------------------------------------
# 2. Metrics
# --------------------------------------------------------------------------
def top1(d):
    if not d["conf"]:
        return None, 0.0
    i = int(np.argmax(d["conf"]))
    return d["cls"][i], float(d["conf"][i])


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_iou(rbs, tbs):
    if not rbs or not tbs:
        return []
    used, out = set(), []
    for rb in rbs:
        best, bj = 0.0, -1
        for j, tb in enumerate(tbs):
            if j in used:
                continue
            v = iou(rb, tb)
            if v > best:
                best, bj = v, j
        if bj >= 0:
            used.add(bj)
            out.append(best)
    return out


def metrics(ref, test):
    top = cnt = danger = miss = extra = wrong = 0
    cds, ious = [], []
    for rd, td in zip(ref, test):
        rc, rconf = top1(rd)
        tc, tconf = top1(td)
        if len(rd["cls"]) == len(td["cls"]):
            cnt += 1
        if bool(DANGER & set(rd["cls"])) == bool(DANGER & set(td["cls"])):
            danger += 1
        if rc is None and tc is None:
            top += 1
        elif rc is None:
            extra += 1
        elif tc is None:
            miss += 1
        else:
            cds.append(abs(rconf - tconf))
            ious.extend(match_iou(rd["xyxy"], td["xyxy"]))
            top += 1 if rc == tc else 0
            wrong += 0 if rc == tc else 1
    return {
        "top1": top / n * 100, "cnt": cnt / n * 100, "danger": danger / n * 100,
        "wrong": wrong, "miss": miss, "extra": extra,
        "mean_cd": float(np.mean(cds)) if cds else 0.0,
        "max_cd": float(np.max(cds)) if cds else 0.0,
        "mean_iou": float(np.mean(ious)) if ious else 1.0,
    }


fp16_m = metrics(fp32_d, fp16_d)
int8_m = metrics(fp32_d, int8_d)


def size_mb(p):
    if os.path.isdir(p):
        return sum(os.path.getsize(os.path.join(dp, f))
                   for dp, _, fs in os.walk(p) for f in fs) / 1e6
    return os.path.getsize(p) / 1e6


pt_mb, ml_mb = size_mb("train14_best.pt"), size_mb("train14_best.mlpackage")

header = ["Variant", "Precision", "Device", "Model_Size_MB", "Speed_ms_per_image",
          "Speedup_vs_FP32", "Top1_Species_Agreement_%", "Detection_Count_Match_%",
          "Danger_Agreement_%", "Wrong_Species_Errors", "Missed_Detections",
          "Extra_Detections", "Mean_Confidence_Diff", "Max_Confidence_Diff",
          "Mean_Box_IoU"]
rows = [
    ["FP32 PyTorch", "FP32", "MPS", round(pt_mb, 1), round(fp32_t/n*1000, 1),
     1.00, 100.0, 100.0, 100.0, 0, 0, 0, 0.0, 0.0, 1.000],
    ["FP16 PyTorch", "FP16", "MPS", round(pt_mb, 1), round(fp16_t/n*1000, 1),
     round(fp32_t/fp16_t, 2), round(fp16_m["top1"], 1), round(fp16_m["cnt"], 1),
     round(fp16_m["danger"], 1), fp16_m["wrong"], fp16_m["miss"], fp16_m["extra"],
     round(fp16_m["mean_cd"], 4), round(fp16_m["max_cd"], 4), round(fp16_m["mean_iou"], 3)],
    ["INT8 CoreML", "INT8", "ANE", round(ml_mb, 1), round(int8_t/n*1000, 1),
     round(fp32_t/int8_t, 2), round(int8_m["top1"], 1), round(int8_m["cnt"], 1),
     round(int8_m["danger"], 1), int8_m["wrong"], int8_m["miss"], int8_m["extra"],
     round(int8_m["mean_cd"], 4), round(int8_m["max_cd"], 4), round(int8_m["mean_iou"], 3)],
]
with open("comparison_metrics.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)
print("\nwrote comparison_metrics.csv")


# --------------------------------------------------------------------------
# 3. Annotated side-by-side images (FP32 | INT8)
# --------------------------------------------------------------------------
def summary(boxes_cls, boxes_conf):
    if not boxes_cls:
        return "no detection"
    return ", ".join(f"{NAMES[c]}:{cf:.2f}" for c, cf in zip(boxes_cls, boxes_conf))


def banner(im, text, color):
    w = im.shape[1]
    bar = np.full((44, w, 3), color, np.uint8)
    cv2.putText(bar, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return cv2.vconcat([bar, im])


def resize_h(im, h=560):
    return cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))


# categorise images: disagreements (one model detects, other doesn't) vs agreements
disagree, agree = [], []
for i, (rd, td) in enumerate(zip(fp32_d, int8_d)):
    rc, _ = top1(rd)
    tc, _ = top1(td)
    if (rc is None) != (tc is None):
        disagree.append(i)
    elif rc is not None and tc is not None:
        agree.append(i)
# show dangerous-animal agreements first
agree.sort(key=lambda i: 0 if (DANGER & set(fp32_d[i]["cls"])) else 1)
selected = [("DIFF", i) for i in disagree] + [("MATCH", i) for i in agree[:10]]

print(f"writing {len(selected)} annotated comparisons to {OUTDIR}/ ...")
for tag, i in selected:
    img = images[i]
    r32 = fp32.predict(img, imgsz=IMGSZ, conf=CONF, classes=CLASSES,
                       device="mps", verbose=False)[0]
    r8 = int8.predict(img, imgsz=IMGSZ, conf=CONF, classes=CLASSES, verbose=False)[0]
    a32 = banner(resize_h(r32.plot()),
                 f"FP32  |  {summary(fp32_d[i]['cls'], fp32_d[i]['conf'])}", (90, 90, 90))
    a8 = banner(resize_h(r8.plot()),
                f"INT8  |  {summary(int8_d[i]['cls'], int8_d[i]['conf'])}", (150, 90, 30))
    combo = cv2.hconcat([a32, a8])
    out = os.path.join(OUTDIR, f"{tag}_{os.path.splitext(os.path.basename(img))[0]}.jpg")
    cv2.imwrite(out, combo)

print(f"done -- {len(selected)} images in {OUTDIR}/")
