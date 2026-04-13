#!/usr/bin/env python3
"""
SmartHandover — Day 4: Train RoBERTa Text-Only + Evaluate on Test Set

Usage:
    python run_day4_train.py                    # Train with defaults (GPU)
    python run_day4_train.py --batch-size 32    # Larger batch
    python run_day4_train.py --eval-only        # Skip training, just evaluate saved model
"""

import argparse
import os
import sys

import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.training.train_text import (
    TextOnlyClassifier, train_model, evaluate_on_test, NUM_CLASSES,
)
from src.data.load_meld import TARGET_LABELS, TARGET_LABEL2ID


def main():
    parser = argparse.ArgumentParser(description="Day 4 — Train RoBERTa Text-Only")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, evaluate saved checkpoint")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join("checkpoints", "roberta_text_only.pt")

    print("=" * 60)
    print("  SmartHandover — Day 4: RoBERTa Text-Only Fine-Tune")
    print(f"  Device: {device}")
    print("=" * 60)

    # --- Train ---
    if not args.eval_only:
        model, history = train_model(
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            lr=args.lr,
            patience=args.patience,
            device=device,
        )

        # Save training history
        hist_df = pd.DataFrame(history)
        os.makedirs("data/processed", exist_ok=True)
        hist_df.to_csv("data/processed/roberta_training_history.csv", index=False)
        print("Training history saved to data/processed/roberta_training_history.csv")

    # --- Load best model ---
    print(f"\nLoading best model from {ckpt_path}...")
    model = TextOnlyClassifier()
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    os.makedirs("data/processed", exist_ok=True)

    # --- Evaluate on validation set (for ensemble grid search) ---
    print("\nEvaluating on MELD validation set...")
    val_df = evaluate_on_test(model, device=device, split="validation")
    val_path = os.path.join("data", "processed", "roberta_val_predictions.csv")
    val_df.to_csv(val_path, index=False)
    print(f"Val predictions saved to: {val_path}")

    # --- Evaluate on test set ---
    print("\nEvaluating on MELD test set...")
    results_df = evaluate_on_test(model, device=device, split="test")

    output_path = os.path.join("data", "processed", "roberta_predictions.csv")
    results_df.to_csv(output_path, index=False)
    print(f"Test predictions saved to: {output_path}")

    # --- Metrics ---
    from src.evaluation.metrics import compute_metrics, print_metrics

    y_true = [TARGET_LABEL2ID[l] for l in results_df["true_label"]]
    y_pred = [TARGET_LABEL2ID[l] for l in results_df["predicted_class"]]
    metrics = compute_metrics(y_true, y_pred, target_names=TARGET_LABELS)

    print("\n" + "=" * 60)
    print("  RoBERTa Text-Only — Test Set Results")
    print("=" * 60)
    print_metrics(metrics)

    # --- Comparison with previous baselines ---
    print("\n" + "=" * 60)
    print("  Baseline Comparison (Days 1-4)")
    print("=" * 60)

    rows = [("RoBERTa (fine-tuned)", metrics["weighted_f1"], metrics["macro_f1"],
             metrics["frustration_recall"])]

    for name, path in [("VADER", "data/processed/vader_predictions.csv"),
                       ("GoEmotions", "data/processed/goemo_predictions.csv")]:
        if os.path.exists(path):
            prev = pd.read_csv(path)
            yt = [TARGET_LABEL2ID[l] for l in prev["true_label"]]
            yp = [TARGET_LABEL2ID[l] for l in prev["predicted_class"]]
            m = compute_metrics(yt, yp, target_names=TARGET_LABELS)
            rows.append((name, m["weighted_f1"], m["macro_f1"], m["frustration_recall"]))

    print(f"\n  {'Model':<22s} {'W-F1':>8s} {'M-F1':>8s} {'Frust-R':>8s}")
    print(f"  {'-'*48}")
    for name, wf1, mf1, fr in sorted(rows, key=lambda x: -x[1]):
        print(f"  {name:<22s} {wf1:>8.4f} {mf1:>8.4f} {fr:>8.4f}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
