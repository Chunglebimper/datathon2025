import argparse, json, torch
from PIL import Image
from transforms import TFMS
from model import MosquitoNet
import numpy as np

"""
eval.py
--------
Evaluates a saved checkpoint on the validation or test split.
Loads the model and dataset, performs forward passes, and reports overall accuracy
and a confusion matrix count.

Usage:
    python src/eval.py --checkpoint runs/best.ckpt --splits configs/splits.json --split val
"""

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--splits", required=True)
ap.add_argument("--split", choices=["val","test"], default="val")
args = ap.parse_args()

ckpt = torch.load(args.checkpoint, map_location="cpu")
classes = ckpt["classes"]; cfg = ckpt.get("cfg", {})
model = MosquitoNet(backbone=cfg.get("backbone","resnet34"), num_classes=len(classes), pretrained=False)
model.load_state_dict(ckpt["model"]); model.eval()

with open(args.splits,"r") as f: s=json.load(f)
pairs = s[args.split]
TF = TFMS(img_size=cfg.get("img_size",448))

right=0
conf = np.zeros((len(classes), len(classes)), dtype=int)
for path,label in pairs:
    x = TF.eval(Image.open(path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        idx = int(torch.argmax(model(x),1).item())
    gt = classes.index(label)
    conf[gt, idx]+=1
    right += (idx==gt)

acc = right/max(1,len(pairs))
print("Accuracy:", acc)
print("Confusion matrix shape:", conf.shape)
