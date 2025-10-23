"""
infer_single.py
----------------
Runs single-image inference using a trained checkpoint.
Prints Top-K predicted species (default K=3) with probabilities.

Usage:
    python src/infer_single.py --image path/to/image.jpg --checkpoint runs/best.ckpt --topk 3
"""
import argparse, torch
from PIL import Image
from transforms import TFMS
from model import MosquitoNet

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--img-size", type=int, default=300)
ap.add_argument("--topk", type=int, default=3)
args = ap.parse_args()

ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False) #updated from 2.6 call
classes = ckpt.get("classes"); cfg = ckpt.get("cfg", {})
model = MosquitoNet(backbone=cfg.get("backbone","resnet34"), num_classes=len(classes), pretrained=False)
model.load_state_dict(ckpt["model"]); model.eval()

TF = TFMS(img_size=args.img_size)
x = TF.eval(Image.open(args.image).convert("RGB")).unsqueeze(0)
with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)[0]
topp = torch.topk(probs, k=min(args.topk, len(classes)))
idxs = topp.indices.tolist(); vals = topp.values.tolist()
print("Top-{} predictions:".format(args.topk))
for i,(ci, pv) in enumerate(zip(idxs, vals), 1):
    print(f"{i}. {classes[ci]}  prob={pv:.3f}")
