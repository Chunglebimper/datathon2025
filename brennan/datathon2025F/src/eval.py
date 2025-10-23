"""
eval.py
--------
Evaluates a saved checkpoint on the validation or test split.
Reports Top-1 and Top-3 accuracy and a confusion matrix size summary.
"""
import argparse, json, torch
from PIL import Image
from transforms import TFMS
from model import MosquitoNet
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--splits", required=True)
ap.add_argument("--split", choices=["val","test"], default="val")
args = ap.parse_args()

ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False) #updated from 2.6 call
classes = ckpt["classes"]; cfg = ckpt.get("cfg", {})
model = MosquitoNet(backbone=cfg.get("backbone","resnet34"), num_classes=len(classes), pretrained=False)
model.load_state_dict(ckpt["model"]); model.eval()

with open(args.splits,"r") as f: s=json.load(f)
pairs = s[args.split]
TF = TFMS(img_size=cfg.get("img_size",300))

right=0; top3_right=0
conf = np.zeros((len(classes), len(classes)), dtype=int)
for path,label in pairs:
    x = TF.eval(Image.open(path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits,1)
        idx = int(torch.argmax(probs,1).item())
        top3 = torch.topk(probs, k=min(3, len(classes)), dim=1).indices[0].tolist()
    gt = classes.index(label)
    conf[gt, idx] += 1
    right += (idx==gt)
    top3_right += (gt in top3)

n = max(1, len(pairs))
print("Accuracy (Top-1):", right/n)
print("Accuracy (Top-3):", top3_right/n)
print("Confusion matrix shape:", conf.shape)
