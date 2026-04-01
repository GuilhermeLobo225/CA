"""
Smart Handover — Evaluation Metrics
Computes classification metrics with focus on frustration detection.
"""

import numpy as np
import torch
from collections import Counter
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data.load_meld import TARGET_LABELS, TARGET_LABEL2ID


def compute_class_weights(dataset):
    """Compute inverse-frequency class weights from a HuggingFace Dataset.

    Formula: weight_c = total_samples / (num_classes * count_c)

    Args:
        dataset: HuggingFace Dataset with 'target_label' column.

    Returns:
        torch.FloatTensor of shape [num_classes], ordered by TARGET_LABELS.
    """
    counts = Counter(dataset["target_label"])
    total = sum(counts.values())
    num_classes = len(TARGET_LABELS)

    weights = []
    for i in range(num_classes):
        c = counts.get(i, 1)  # avoid division by zero
        weights.append(total / (num_classes * c))

    return torch.FloatTensor(weights)


def compute_metrics(y_true, y_pred, target_names=None):
    """Compute full evaluation metrics.

    Args:
        y_true: list/array of ground truth label indices.
        y_pred: list/array of predicted label indices.
        target_names: list of class name strings.

    Returns:
        dict with:
          - weighted_f1, macro_f1, accuracy
          - per_class: {class_name: {precision, recall, f1, support}}
          - frustration_recall (key metric, extracted for convenience)
          - confusion_matrix: numpy array [num_classes, num_classes]
    """
    if target_names is None:
        target_names = TARGET_LABELS

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Sklearn classification report as dict
    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    # Per-class metrics
    per_class = {}
    for name in target_names:
        if name in report:
            per_class[name] = {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": int(report[name]["support"]),
            }

    # Frustration recall (the key metric for early stopping)
    frustration_recall = per_class.get("frustration", {}).get("recall", 0.0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(target_names))))

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "per_class": per_class,
        "frustration_recall": frustration_recall,
        "confusion_matrix": cm,
    }


def print_metrics(metrics_dict):
    """Pretty-print evaluation metrics to console."""
    print(f"\n{'='*60}")
    print(f"  Accuracy     : {metrics_dict['accuracy']:.4f}")
    print(f"  Weighted F1  : {metrics_dict['weighted_f1']:.4f}")
    print(f"  Macro F1     : {metrics_dict['macro_f1']:.4f}")
    print(f"  Frustration R: {metrics_dict['frustration_recall']:.4f}  << KEY METRIC")
    print(f"{'='*60}")

    print(f"\n  {'Class':<15s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'Support':>8s}")
    print(f"  {'-'*43}")
    for name, vals in metrics_dict["per_class"].items():
        marker = " <<" if name == "frustration" else ""
        print(
            f"  {name:<15s} {vals['precision']:>6.3f} {vals['recall']:>6.3f} "
            f"{vals['f1']:>6.3f} {vals['support']:>8d}{marker}"
        )

    print_confusion_matrix(metrics_dict["confusion_matrix"])


def print_confusion_matrix(cm, labels=None):
    """Print a formatted confusion matrix."""
    if labels is None:
        labels = TARGET_LABELS

    # Abbreviated labels for display
    short = [l[:5] for l in labels]

    print(f"\n  Confusion Matrix (rows=true, cols=pred):")
    header = "  " + " " * 8 + "".join(f"{s:>7s}" for s in short)
    print(header)
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>7d}" for v in row)
        print(f"  {short[i]:>7s} {row_str}")
