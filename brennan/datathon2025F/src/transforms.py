"""
transforms.py
--------------
Image preprocessing and augmentation pipelines.
Train: RandomResizedCrop(300), flip, light color jitter, normalize, RandomErasing.
Val: Resize+CenterCrop(300), normalize.
"""
import torchvision.transforms as T

class TFMS:
    def __init__(self, img_size=300, aug_cfg=None):
        aug_cfg = aug_cfg or {}   # ← add this line
        cj = aug_cfg.get("color_jitter", [0.1,0.1,0.1,0.05])
        scale = aug_cfg.get("random_resized_crop_scale", [0.8,1.0])
        self.train = T.Compose([
            T.RandomResizedCrop(img_size, scale=tuple(scale)),
            T.RandomHorizontalFlip(p=aug_cfg.get("hflip_p", 0.5)),
            T.ColorJitter(*cj),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            T.RandomErasing(p=aug_cfg.get("rand_erase_p", 0.15))
        ])
        self.eval = T.Compose([
            T.Resize(int(img_size*1.15)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])