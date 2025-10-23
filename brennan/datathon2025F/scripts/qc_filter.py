"""
qc_filter.py
-------------
Lists basic quality metrics to help you exclude very blurry or tiny images before training.
This does NOT delete files; it writes a CSV with (path, ok flag, width, height, Laplacian variance).

Usage:
    python scripts/qc_filter.py --root "<DATA_ROOT>" --min-side 120 --blur-th 80 --out-csv configs/filtered_index.csv
"""
import argparse, csv
from pathlib import Path
import numpy as np
import cv2

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)              # DATA_ROOT
    ap.add_argument("--min-side", type=int, default=120)  # reject tiny images
    ap.add_argument("--blur-th", type=float, default=80.0)# tune: 60–100 typical
    ap.add_argument("--out-csv", default="configs/filtered_index.csv")
    args = ap.parse_args()

    rows = [("path","ok","w","h","lap_var")]
    for p in Path(args.root).rglob("*"):
        if p.suffix.lower() not in IMG_EXTS: continue
        img = cv2.imdecode(np.fromfile(str(p), dtype='uint8'), cv2.IMREAD_COLOR)
        if img is None:
            rows.append((str(p),0,0,0,0)); continue
        h,w = img.shape[:2]
        ok = 1
        if min(h,w) < args.min_side: ok = 0
        lv = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if lv < args.blur_th: ok = 0
        rows.append((str(p),ok,w,h,float(lv)))

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"Wrote {args.out_csv}. Inspect rows with ok=0 to see rejections.")
