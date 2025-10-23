# export_torchscript.py
import torch
from src.model import MosquitoNet
ckpt = torch.load(r"C:\Users\bjren\PycharmProjects\datathon2025F\runs\best.ckpt",
                  map_location="cpu", weights_only=False) #update this
model = MosquitoNet(backbone=ckpt["cfg"].get("backbone","resnet34"),
                    num_classes=len(ckpt["classes"]), pretrained=False)
model.load_state_dict(ckpt["model"]); model.eval()
dummy = torch.randn(1,3,300,300)
torch.jit.trace(model, dummy).save("mosquito_resnet34_scripted.pt")
with open("classes.txt","w") as f: f.write("\n".join(ckpt["classes"]))
print("Saved mosquito_resnet34_scripted.pt and classes.txt")
