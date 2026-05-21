"""
Compare FP32 vs FP16 vs INT8 on the TBL-1-12-2023--29-01-2024 image set.

Speed    : wall-clock inference time per image, averaged over the folder.
Accuracy : the folder has no ground-truth labels, so the FP32 PyTorch model is
           used as the reference. FP16 and INT8 are scored on how closely they
           agree with FP32 (detection count, top-1 class, confidence, box IoU).
"""

import glob
import os
import time

import numpy as np
from ultralytics import YOLO

FOLDER = "TBL-1-12-2023--29-01-2024"
IMGSZ, CONF = 800, 0.6
NAMES = {0: 'Leopard', 1: 'Cat', 2: 'Dog', 3: 'Deer', 4: 'Goat', 5: 'Monkey',
         6: 'Tiger', 7: 'Wild_boar', 8: 'Cow', 9: 'Hen', 10: 'Bear', 11: 'Byson'}
DANGER = {0, 6, 10}  # Leopard, Tiger, Bear

images = sorted(glob.glob(os.path.join(FOLDER, "*.jpg")))
print(f"{len(images)} images\n")

fp32 = YOLO("train14_best.pt")
int8 = YOLO("train14_best.mlpackage")


def run_all(model, **kw):
    """Run model over every image; return list of detections + total seconds."""
    model.predict(images[0], imgsz=IMGSZ, conf=CONF, verbose=False, **kw)  # warmup
    dets, t0 = [], time.perf_counter()
    for img in images:
        b = model.predict(img, imgsz=IMGSZ, conf=CONF, verbose=False, **kw)[0].boxes
        dets.append({
            "cls": b.cls.tolist(),
            "conf": b.conf.tolist(),
            "xyxy": b.xyxy.tolist(),
        })
    return dets, time.perf_counter() - t0


def top1(d):
    """Highest-confidence detection: (class, conf, box) or (None, 0, None)."""
    if not d["conf"]:
        return None, 0.0, None
    i = int(np.argmax(d["conf"]))
    return int(d["cls"][i]), float(d["conf"][i]), d["xyxy"][i]


def iou(a, b):
    if a is None or b is None:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_iou(ref_boxes, test_boxes):
    """Greedily match every ref box to its best test box; return their IoUs."""
    if not ref_boxes or not test_boxes:
        return []
    used, ious = set(), []
    for rb in ref_boxes:
        best, best_j = 0.0, -1
        for j, tb in enumerate(test_boxes):
            if j in used:
                continue
            v = iou(rb, tb)
            if v > best:
                best, best_j = v, j
        if best_j >= 0:
            used.add(best_j)
            ious.append(best)
    return ious


# ---- run all three ----
fp32_d, fp32_t = run_all(fp32, device="mps")
fp16_d, fp16_t = run_all(fp32, device="mps", half=True)
int8_d, int8_t = run_all(int8)

n = len(images)
print("================  SPEED  ================")
print(f"{'FP32 PyTorch/MPS':22s}: {fp32_t/n*1000:7.1f} ms/img   ({fp32_t:5.1f}s total)")
print(f"{'FP16 PyTorch/MPS':22s}: {fp16_t/n*1000:7.1f} ms/img   ({fp16_t:5.1f}s total)")
print(f"{'INT8 CoreML/ANE':22s}: {int8_t/n*1000:7.1f} ms/img   ({int8_t:5.1f}s total)")
print(f"INT8 speedup vs FP32  : {fp32_t/int8_t:.2f}x faster\n")


def accuracy(ref, test, label):
    count_match = top_match = both_empty = miss = extra = danger_match = 0
    conf_deltas, ious, disagreements = [], [], []
    for img, rd, td in zip(images, ref, test):
        rc, rconf, rbox = top1(rd)
        tc, tconf, tbox = top1(td)
        if len(rd["cls"]) == len(td["cls"]):
            count_match += 1
        # dangerous-animal presence agreement
        r_danger = bool(DANGER & set(int(c) for c in rd["cls"]))
        t_danger = bool(DANGER & set(int(c) for c in td["cls"]))
        if r_danger == t_danger:
            danger_match += 1
        # top-1 agreement
        if rc is None and tc is None:
            both_empty += 1
            top_match += 1
        elif rc is None:
            extra += 1
        elif tc is None:
            miss += 1
        else:
            conf_deltas.append(abs(rconf - tconf))
            ious.extend(match_iou(rd["xyxy"], td["xyxy"]))
            if rc == tc:
                top_match += 1
            else:
                disagreements.append((os.path.basename(img), NAMES[rc], NAMES[tc]))

    print(f"========  ACCURACY: {label} vs FP32 (reference)  ========")
    print(f"Top-1 class agreement   : {top_match}/{n}  ({top_match/n*100:.1f}%)")
    print(f"Detection-count match   : {count_match}/{n}  ({count_match/n*100:.1f}%)")
    print(f"Danger (Leo/Tiger/Bear) : {danger_match}/{n}  ({danger_match/n*100:.1f}%) agreement")
    print(f"Both found nothing      : {both_empty}/{n}")
    print(f"FP32 found, {label} missed : {miss}")
    print(f"{label} found, FP32 missed : {extra}")
    if conf_deltas:
        print(f"Confidence diff (matched): mean {np.mean(conf_deltas):.4f}  max {np.max(conf_deltas):.4f}")
    if ious:
        print(f"Box IoU (matched)        : mean {np.mean(ious):.4f}  min {np.min(ious):.4f}")
    if disagreements:
        print(f"Top-1 disagreements ({len(disagreements)}):")
        for fn, a, b in disagreements[:10]:
            print(f"   {fn}: FP32={a}  {label}={b}")
    print()


accuracy(fp32_d, fp16_d, "FP16")
accuracy(fp32_d, int8_d, "INT8")
