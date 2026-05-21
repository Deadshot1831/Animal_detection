"""
Merge the per-folder FP32 (train14_best.pt) vs INT8 results into one report.

Inputs : per_image_results.csv, per_image_results_New_images.csv
Outputs: combined_comparison.csv  -- 206 rows, one per image, with a Folder column
         combined_summary.txt     -- overall agreement, speed, per-species, disagreements
"""

import csv
from collections import Counter

SOURCES = [
    ("TBL-1-12-2023--29-01-2024", "per_image_results.csv"),
    ("New_images", "per_image_results_New_images.csv"),
]
SPECIES = ["Leopard", "Tiger", "Bear", "Nothing"]

# ---- load + merge ----
rows = []
for folder, path in SOURCES:
    for r in csv.DictReader(open(path)):
        rows.append({"Folder": folder, **r})

fields = ["Folder", "Image", "FP32_Detections", "FP32_Top_Species", "FP32_Top_Confidence",
          "INT8_Detections", "INT8_Top_Species", "INT8_Top_Confidence",
          "Species_Agreement", "FP32_Time_ms", "INT8_Time_ms"]
with open("combined_comparison.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

# ---- stats ----
n = len(rows)
agree = sum(1 for r in rows if r["Species_Agreement"] == "Yes")
fp32_t = sum(float(r["FP32_Time_ms"]) for r in rows) / n
int8_t = sum(float(r["INT8_Time_ms"]) for r in rows) / n

# confidence delta where both detect the SAME species (real animal)
conf_deltas = [abs(float(r["FP32_Top_Confidence"]) - float(r["INT8_Top_Confidence"]))
               for r in rows
               if r["Species_Agreement"] == "Yes" and r["FP32_Top_Species"] != "Nothing"]

fp32_sp = Counter(r["FP32_Top_Species"] for r in rows)
int8_sp = Counter(r["INT8_Top_Species"] for r in rows)
disagreements = [r for r in rows if r["Species_Agreement"] == "No"]

# ---- build report ----
L = []
L.append("=" * 64)
L.append("  COMBINED REPORT -- train14_best.pt (FP32) vs INT8")
L.append("=" * 64)
_folders = ", ".join(f"{f}={c}" for f, c in Counter(r["Folder"] for r in rows).items())
L.append(f"Total images          : {n}  ({_folders})")
L.append("")
L.append("-- SPEED --")
L.append(f"FP32 (train14_best.pt): {fp32_t:6.1f} ms/image")
L.append(f"INT8 (CoreML)         : {int8_t:6.1f} ms/image")
L.append(f"INT8 speedup          : {fp32_t/int8_t:.2f}x faster")
L.append("")
L.append("-- AGREEMENT --")
L.append(f"Species agreement     : {agree}/{n}  ({agree/n*100:.1f}%)")
if conf_deltas:
    L.append(f"Confidence difference : mean {sum(conf_deltas)/len(conf_deltas):.4f}  "
             f"max {max(conf_deltas):.4f}   (on images where both agree)")
L.append("")
L.append("-- PER-SPECIES (top-1 detection per image) --")
L.append(f"{'Species':10s} {'FP32':>6s} {'INT8':>6s}   {'FP32->INT8 agreement':>22s}")
for sp in SPECIES:
    same = sum(1 for r in rows if r["FP32_Top_Species"] == sp
               and r["INT8_Top_Species"] == sp)
    base = fp32_sp.get(sp, 0)
    pct = f"{same}/{base} ({same/base*100:.0f}%)" if base else "-"
    L.append(f"{sp:10s} {fp32_sp.get(sp,0):6d} {int8_sp.get(sp,0):6d}   {pct:>22s}")
L.append("")
L.append(f"-- DISAGREEMENTS ({len(disagreements)}) --")
for r in disagreements:
    L.append(f"  [{r['Folder']}] {r['Image']}")
    L.append(f"      FP32: {r['FP32_Top_Species']} {r['FP32_Top_Confidence']}   "
             f"INT8: {r['INT8_Top_Species']} {r['INT8_Top_Confidence']}")
L.append("=" * 64)

report = "\n".join(L)
print(report)
with open("combined_summary.txt", "w") as f:
    f.write(report + "\n")
print(f"\nwrote combined_comparison.csv ({n} rows)")
print("wrote combined_summary.txt")
