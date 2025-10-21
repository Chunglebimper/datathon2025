import argparse, json
from pathlib import Path
from sklearn.model_selection import train_test_split

"""
make_splits.py
---------------
Creates stratified train/validation splits from the dataset folder structure.
Automatically detects subfolders under each species directory
and writes configs/splits.json for training.

Usage:
    python scripts/make_splits.py --data-root "<DATA_ROOT>" --val-perc 0.15 --out configs/splits.json
    see readMe.
"""


IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}

def scan(data_root: Path):
    samples = []
    for species_dir in sorted([d for d in data_root.iterdir() if d.is_dir()]):
        label = species_dir.name
        # tolerate different subfolder names
        for sub in ["Jpeg","JPEG","jpeg","jpg","images","Image","image",""]:
            jpeg_dir = species_dir / sub if sub else species_dir
            if jpeg_dir.exists():
                added = False
                for p in jpeg_dir.rglob("*"):
                    if p.suffix.lower() in IMG_EXTS:
                        samples.append((str(p.resolve()), label))
                        added = True
                if added:
                    break
    return samples

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--val-perc", type=float, default=0.15)
    ap.add_argument("--out", default="configs/splits.json")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    pairs = scan(data_root)
    if not pairs:
        raise SystemExit(f"No images found under {data_root}. Check your path and 'Jpeg/' subfolders.")

    X = [p for p,_ in pairs]
    y = [l for _,l in pairs]

    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=args.val_perc, random_state=42, stratify=y)

    labels = sorted(set(y))
    out = {"labels": labels, "train": list(zip(Xtr,ytr)), "val": list(zip(Xval,yval)), "test": []}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}: {len(Xtr)} train / {len(Xval)} val / 0 test.")
