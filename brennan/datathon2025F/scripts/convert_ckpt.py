# scripts/convert_ckpt.py
import torch
from src.model import MosquitoNet

src = "runs/best.ckpt"
dst_ckpt = "runs/best_v2.ckpt"
dst_state = "runs/best_model.pt"
dst_scripted = "runs/mosquito_resnet34_scripted.pt"

# 1) Load old ckpt with explicit weights_only=False (trusted file)
ckpt = torch.load(src, map_location="cpu", weights_only=False)

# 2) Re-save the checkpoint in a way PyTorch 2.6+ likes
torch.save(ckpt, dst_ckpt)
print(f"Saved converted checkpoint -> {dst_ckpt}")

# 3) Save weights-only .pt (state_dict)
torch.save(ckpt["model"], dst_state)
print(f"Saved weights-only model -> {dst_state}")

# 4) Optional: make a TorchScript file for deployment
classes = ckpt["classes"]
model = MosquitoNet(backbone=ckpt["cfg"].get("backbone","resnet34"), num_classes=len(classes), pretrained=False)
model.load_state_dict(ckpt["model"]); model.eval()
dummy = torch.randn(1,3,300,300)
traced = torch.jit.trace(model, dummy)
traced.save(dst_scripted)
print(f"Saved TorchScript model -> {dst_scripted}")
