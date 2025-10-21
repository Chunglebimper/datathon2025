# Optional: square-pad + resize images to a new folder on disk.
# Not necessary for training (transforms already handle resizing).
import argparse
from pathlib import Path
from PIL import Image, ImageOps

"""
optional_preprocess.py
-----------------------
Optional preprocessing utility for square-padding and resizing all images to a fixed size.
Not required for training (transforms already handle resizing dynamically).
Use only if you want pre-processed copies saved on disk.

Usage:
    python scripts/optional_preprocess.py --in-root "<RAW_DATA>" --out-root "<OUT_DIR>" --size 448
"""

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--size", type=int, default=448)
    args = ap.parse_args()

    in_root, out_root = Path(args.in_root), Path(args.out_root)
    for p in in_root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            rel = p.relative_to(in_root)
            out_p = out_root / rel
            out_p.parent.mkdir(parents=True, exist_ok=True)
            img = Image.open(p).convert("RGB")
            img = ImageOps.exif_transpose(img)
            max_side = max(img.size)
            square = Image.new("RGB", (max_side, max_side), (255,255,255))
            ox = (max_side - img.size[0]) // 2
            oy = (max_side - img.size[1]) // 2
            square.paste(img, (ox, oy))
            square = square.resize((args.size, args.size), Image.BICUBIC)
            square.save(out_p.with_suffix(".jpg"), quality=95)
