import argparse, torch
from PIL import Image
from transforms import TFMS
from model import MosquitoNet

"""
infer_single.py
----------------
Runs single-image inference using a trained checkpoint.
Loads one mosquito image, applies the same preprocessing as validation, 
and prints the predicted species and confidence score.

Usage:
    python src/infer_single.py --image path/to/image.jpg --checkpoint runs/best.ckpt
"""

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--img-size", type=int, default=448)
args = ap.parse_args()

ckpt = torch.load(args.checkpoint, map_location="cpu")
classes = ckpt.get("classes"); cfg = ckpt.get("cfg", {})
model = MosquitoNet(backbone=cfg.get("backbone","resnet34"), num_classes=len(classes), pretrained=False)
model.load_state_dict(ckpt["model"]); model.eval()

TF = TFMS(img_size=args.img_size)
img = Image.open(args.image).convert("RGB")
x = TF.eval(img).unsqueeze(0)
with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)[0]
idx = int(torch.argmax(probs).item())
print(f"Prediction: {classes[idx]}  prob={probs[idx].item():.3f}")
