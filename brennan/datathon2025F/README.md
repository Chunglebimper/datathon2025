# Datathon 2025F: Mosquito-ID 
# Authored by Brennan Miller week of Oct 19th
# Last updated 10/24

Single-image mosquito **species classification** with PyTorch (ResNet-34 by default). 
Assumes your dataset has roughly 21 species; each species folder contains a `Jpeg/` subfolder with images like `Aedes aegypti/Jpeg/IMG_7334.jpg`
Does not account for preprocessing into individual mosquito images at the moment

## Folder layout
```
mosquito-id/
├─ README.md
├─ requirements.txt
├─ .env.example
├─ configs/
│  └─ default.yaml
├─ src/
│  ├─ train.py
│  ├─ data.py
│  ├─ transforms.py
│  ├─ model.py
│  ├─ utils.py
│  ├─ eval.py
│  └─ infer_single.py
└─ scripts/
   ├─ make_splits.py
   ├─ sanity_check.py
   └─ optional_preprocess.py
```

## Quickstart 
pip install -r requirements.txt
Set `DATA_ROOT` to your dataset folder in env file.

## Create stratified splits:
   ```bash
   python scripts/make_splits.py --data-root "<PATH TO YOUR DATA>" --val-perc 0.15 --out configs/splits.json
   python scripts/sanity_check.py
   ```
## Train (CUDA is used automatically if available):
   ```bash
   python src/train.py --config configs/default.yaml
   ```
## Evaluate
   ```bash
   python src/eval.py --checkpoint runs/best.ckpt --splits configs/splits.json --split val
   python src/infer_single.py --image "<some_val_image.jpg>" --checkpoint runs/best.ckpt --topk 3
   ```
