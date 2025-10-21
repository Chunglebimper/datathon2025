import torchvision.transforms as T

"""
transforms.py
--------------
Contains image preprocessing and augmentation pipelines.
Applies resizing, random crops, flips, color jitter, normalization, and random erasing.
Used by both training and validation DataLoaders.
We may or may not end up using the transforms but was useful for my testing.
"""

class TFMS:
    def __init__(self, img_size=448, aug_cfg=None):
        cj = aug_cfg.get("color_jitter", [0.2,0.2,0.2,0.1]) if aug_cfg else [0.2,0.2,0.2,0.1]
        scale = aug_cfg.get("random_resized_crop_scale", [0.7,1.0]) if aug_cfg else [0.7,1.0]
        self.train = T.Compose([
            T.RandomResizedCrop(img_size, scale=tuple(scale)),
            T.RandomHorizontalFlip(p=aug_cfg.get("hflip_p",0.5) if aug_cfg else 0.5),
            T.ColorJitter(*cj),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            T.RandomErasing(p=aug_cfg.get("rand_erase_p",0.25))
        ])
        self.eval = T.Compose([
            T.Resize(int(img_size*1.15)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
