import argparse, os, yaml
from pathlib import Path
import torch, torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from data import build_loaders
from transforms import TFMS
from model import MosquitoNet
from utils import EarlyStopper, macro_f1_from_logits

"""
train.py
---------
Main training script for mosquito species classification.
Loads the dataset splits, applies augmentations, initializes the model,
and trains using CrossEntropy loss with early stopping based on macro-F1.

Outputs:
- Trained weights saved as runs/best.ckpt
- Console log with validation accuracy and macro-F1 per epoch
"""

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--splits", default=None)
    return ap.parse_args()

def load_cfg(path):
    with open(path, "r") as f: cfg = yaml.safe_load(f)
    if cfg.get("data_root") is None: cfg["data_root"] = os.getenv("DATA_ROOT")
    return cfg

if __name__ == "__main__":
    args = parse_args(); cfg = load_cfg(args.config)
    out_dir = Path(cfg.get("output_dir","runs")); out_dir.mkdir(parents=True, exist_ok=True)

    tfms = TFMS(img_size=cfg.get("img_size",448), aug_cfg=cfg.get("aug",{}))
    train_loader, val_loader, meta = build_loaders(cfg.get("splits_file"), cfg["img_size"], cfg["batch_size"], cfg["num_workers"], cfg.get("balance_sampler",True), tfms)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MosquitoNet(backbone=cfg.get("backbone","resnet34"), num_classes=meta["num_classes"], pretrained=cfg.get("pretrained",True)).to(device)

    crit = nn.CrossEntropyLoss(); opt = AdamW(model.parameters(), lr=cfg.get("lr",3e-4), weight_decay=cfg.get("weight_decay",1e-4))
    sch = CosineAnnealingLR(opt, T_max=cfg.get("epochs",25))
    stopper = EarlyStopper(patience=cfg.get("patience",7))

    best_f1 = -1; best_path = out_dir/"best.ckpt"
    for epoch in range(cfg.get("epochs",25)):
        model.train(); running=0
        for xb,yb in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            xb,yb = xb.to(device), yb.to(device)
            opt.zero_grad(); logits = model(xb); loss = crit(logits,yb); loss.backward(); opt.step(); running += loss.item()*xb.size(0)
        sch.step()
        # val
        model.eval(); correct=0; count=0; f1s=[]
        with torch.no_grad():
            for xb,yb in val_loader:
                xb,yb = xb.to(device), yb.to(device)
                logits = model(xb)
                preds = torch.argmax(logits,1)
                correct += (preds==yb).sum().item(); count += yb.numel()
                f1s.append(macro_f1_from_logits(logits,yb, meta["num_classes"]))
        acc = correct/max(count,1); f1 = sum(f1s)/max(len(f1s),1)
        print(f"Val acc={acc:.4f} macroF1={f1:.4f}")
        if stopper.step(f1):
            torch.save({"model": model.state_dict(), "classes": meta["classes"], "cfg": cfg}, best_path)
            best_f1 = f1; print(f"Saved best -> {best_path}")
        if stopper.should_stop: print("Early stopping."); break
    print(f"Best macro-F1: {best_f1:.4f} at {best_path}")
