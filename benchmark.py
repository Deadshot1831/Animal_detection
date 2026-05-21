"""
Benchmark: OLD approach vs OPTIMISED approach for train14_best.pt.

OLD      = model reloaded from disk on every call  + CPU inference
OPTIMISED= model loaded once                       + MPS (GPU) inference

Run:  python3 benchmark.py
"""

import time
import torch
from ultralytics import YOLO
from ultralytics.utils import ASSETS

IMG = str(ASSETS / "bus.jpg")
MODEL_PATH = "train14_best.pt"
N = 10  # timed iterations


def avg(fn, n=N):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


print(f"Device: MPS available = {torch.backends.mps.is_available()}\n")

# --- Isolate model LOAD time (the cost paid on every call in the old code) ---
load_time = avg(lambda: YOLO(MODEL_PATH), n=5)
print(f"Model load from disk        : {load_time*1000:8.1f} ms")

# --- Pure INFERENCE time on CPU (load excluded) ---
m_cpu = YOLO(MODEL_PATH)
m_cpu.predict(IMG, imgsz=800, conf=0.6, device="cpu", verbose=False)  # warmup
infer_cpu = avg(lambda: m_cpu.predict(IMG, imgsz=800, conf=0.6, device="cpu", verbose=False))
print(f"Inference only (CPU)        : {infer_cpu*1000:8.1f} ms")

# --- Pure INFERENCE time on MPS (load excluded) ---
m_mps = YOLO(MODEL_PATH)
m_mps.predict(IMG, imgsz=800, conf=0.6, device="mps", verbose=False)  # warmup
infer_mps = avg(lambda: m_mps.predict(IMG, imgsz=800, conf=0.6, device="mps", verbose=False))
print(f"Inference only (MPS / GPU)  : {infer_mps*1000:8.1f} ms")

# --- Realistic OLD path: fresh model + first predict every call ---
# (a fresh model pays an extra fuse/warmup cost on its very first predict)
def old_call():
    m = YOLO(MODEL_PATH)
    m.predict(IMG, imgsz=800, conf=0.6, device="cpu", verbose=False)

old_per_call = avg(old_call, n=5)
print(f"OLD realistic (reload+1st infer): {old_per_call*1000:6.1f} ms")

# --- End-to-end per call ---
new_per_call = infer_mps                      # load once + MPS (optimised code)

print("\n" + "=" * 48)
print(f"OLD  per call (reload + CPU): {old_per_call*1000:8.1f} ms")
print(f"NEW  per call (cached + MPS): {new_per_call*1000:8.1f} ms")
print(f"Speedup                    : {old_per_call/new_per_call:8.1f}x faster")
print("=" * 48)
