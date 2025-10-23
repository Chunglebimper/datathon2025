"""
utils.py
---------
Utility functions for training:
- EarlyStopper: stops training when validation metric plateaus.
- macro_f1_from_logits: macro-averaged F1 from logits via torchmetrics.
"""
import torch
from torchmetrics.functional import f1_score

class EarlyStopper:
    def __init__(self, patience=7):
        self.patience = patience; self.best=None; self.count=0
    def step(self, metric):
        if self.best is None or metric > self.best:
            self.best = metric; self.count=0; return True
        self.count += 1; return False
    @property
    def should_stop(self): return self.count >= self.patience

def macro_f1_from_logits(logits, targets, num_classes):
    preds = torch.argmax(logits, dim=1)
    return f1_score(preds, targets, task="multiclass", num_classes=num_classes, average="macro").item()
