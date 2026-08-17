
from __future__ import annotations
from typing import Dict, Any
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

def compute_metrics(y_true, y_pred, labels):
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {"macro_f1": macro_f1, "micro_f1": micro_f1, "accuracy": acc, "report": report, "confusion_matrix": cm}
