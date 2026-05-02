#!/usr/bin/env python3
"""
SmartHandover — Day 5: Text Ensemble (RoBERTa + GoEmotions + VADER)

Grid search on VALIDATION set, evaluate on TEST set.

Usage:
    python run_day5_ensemble.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classifiers.ensemble import (
    TextEnsembleClassifier, vader_to_probs, goemo_row_to_probs,
    TARGET_LABELS, TARGET_LABEL2ID,
)
from src.evaluation.metrics import compute_metrics, print_metrics


def load_and_prepare(roberta_path):
    """Load prediction CSVs and build aligned probability matrices.

    Merges on (text, true_label) to guarantee row alignment across all 3 models.

    Args:
        roberta_path: Path to the RoBERTa predictions CSV for this split.

    Returns:
        roberta_probs [N,5], goemo_probs [N,5], vader_probs [N,5], true_labels [N]
    """
    roberta_df = pd.read_csv(roberta_path)
    goemo_df = pd.read_csv("data/processed/goemo_predictions.csv")
    vader_df = pd.read_csv("data/processed/vader_predictions.csv")

    # Keep order via temporary index
    roberta_df["_order"] = range(len(roberta_df))

    # Select only needed columns for merge
    goemo_prob_cols = [c for c in goemo_df.columns if c.startswith("goemo_")]
    goemo_sub = goemo_df[["text", "true_label"] + goemo_prob_cols]

    vader_sub = vader_df[["text", "true_label", "vader_compound"]]

    # Merge on (text, true_label) — much safer than text alone
    merged = roberta_df.merge(goemo_sub, on=["text", "true_label"], how="inner")
    merged = merged.merge(vader_sub, on=["text", "true_label"], how="inner")

    # Handle potential duplicates: keep first per original roberta order
    merged = merged.drop_duplicates(subset="_order").sort_values("_order").reset_index(drop=True)

    n_matched = len(merged)
    n_total = len(roberta_df)
    if n_matched < n_total:
        print(f"  [WARN] Only {n_matched}/{n_total} samples aligned across all 3 models")
    else:
        print(f"  Aligned {n_matched} samples OK")

    # Extract probability matrices
    rob_prob_cols = sorted([c for c in roberta_df.columns if c.startswith("prob_")])
    roberta_probs = merged[rob_prob_cols].values

    goemo_probs = np.array([goemo_row_to_probs(row) for _, row in merged.iterrows()])
    vader_probs = np.array([vader_to_probs(row["vader_compound"]) for _, row in merged.iterrows()])

    true_labels = [TARGET_LABEL2ID[l] for l in merged["true_label"]]

    return roberta_probs, goemo_probs, vader_probs, true_labels


def main():
    print("=" * 60)
    print("  SmartHandover — Day 5: Text Ensemble")
    print("=" * 60)

    # Check required CSVs
    required = {
        "RoBERTa (test)": "data/processed/roberta_predictions.csv",
        "RoBERTa (val)": "data/processed/roberta_val_predictions.csv",
        "GoEmotions": "data/processed/goemo_predictions.csv",
        "VADER": "data/processed/vader_predictions.csv",
    }
    missing = [name for name, path in required.items() if not os.path.exists(path)]
    if missing:
        print(f"\n[ERROR] Missing CSVs: {', '.join(missing)}")
        print("Run Days 1-4 first (python run_day4_train.py generates RoBERTa CSVs).")
        return

    # ---------------------------------------------------------------
    # Step 1: Grid search on VALIDATION set
    # ---------------------------------------------------------------
    print("\n--- Grid Search on Validation Set ---")
    val_rob, val_goemo, val_vader, val_labels = load_and_prepare(
        "data/processed/roberta_val_predictions.csv"
    )

    ensemble = TextEnsembleClassifier()
    best = ensemble.find_best_weights(val_rob, val_goemo, val_vader, val_labels)

    print(f"  Best weights: alpha(RoBERTa)={best['alpha']}, "
          f"beta(GoEmo)={best['beta']}, gamma(VADER)={best['gamma']}")
    print(f"  Val W-F1: {best['weighted_f1']:.4f}")

    # ---------------------------------------------------------------
    # Step 2: Evaluate on TEST set with best weights
    # ---------------------------------------------------------------
    print("\n--- Evaluating on Test Set ---")
    test_rob, test_goemo, test_vader, test_labels = load_and_prepare(
        "data/processed/roberta_predictions.csv"
    )

    combined = (ensemble.alpha * test_rob +
                ensemble.beta * test_goemo +
                ensemble.gamma * test_vader)
    ensemble_preds = np.argmax(combined, axis=-1)

    metrics = compute_metrics(test_labels, ensemble_preds, target_names=TARGET_LABELS)

    print("\n" + "=" * 60)
    print("  Text Ensemble — Test Set Results")
    print("=" * 60)
    print_metrics(metrics)

    # ---------------------------------------------------------------
    # Step 3: Full comparison table
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Full Comparison: Individual Models vs Ensemble")
    print("=" * 60)

    all_results = [("Text Ensemble", metrics["weighted_f1"], metrics["macro_f1"],
                    metrics["frustration_recall"])]

    for name, probs in [("RoBERTa (fine-tuned)", test_rob),
                        ("GoEmotions", test_goemo),
                        ("VADER", test_vader)]:
        preds = np.argmax(probs, axis=-1)
        m = compute_metrics(test_labels, preds.tolist(), target_names=TARGET_LABELS)
        all_results.append((name, m["weighted_f1"], m["macro_f1"], m["frustration_recall"]))

    print(f"\n  {'Model':<22s} {'W-F1':>8s} {'M-F1':>8s} {'Frust-R':>8s}")
    print(f"  {'-'*48}")
    for name, wf1, mf1, fr in sorted(all_results, key=lambda x: -x[1]):
        marker = " <-- BEST" if name == "Text Ensemble" else ""
        print(f"  {name:<22s} {wf1:>8.4f} {mf1:>8.4f} {fr:>8.4f}{marker}")

    print(f"\n  Ensemble weights (tuned on val): "
          f"RoBERTa={ensemble.alpha}, GoEmo={ensemble.beta}, VADER={ensemble.gamma}")
    print("Done.")


if __name__ == "__main__":
    main()
