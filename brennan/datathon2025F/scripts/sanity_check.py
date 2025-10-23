"""
sanity_check.py
----------------
Quick utility to verify the generated splits.json.
Prints per-class counts and confirms number of classes in the dataset.
Run this after make_splits.py to confirm the split worked correctly.
"""
import json, collections
with open("configs/splits.json","r") as f:
    s = json.load(f)
ctr = collections.Counter([lbl for _,lbl in s["train"]])
print("Class counts (train):")
for k,v in ctr.most_common():
    print(f"{k}: {v}")
print("\nNum classes:", len(s["labels"])) 
