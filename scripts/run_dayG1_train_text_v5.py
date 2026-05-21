#!/usr/bin/env python3
"""SmartHandover - Day G1: Re-fine-tune RoBERTa with focal loss (v5).

Mirrors the v4 ``combined_cw`` condition (MELD train + filtered synthetic
text + inverse-frequency class weights) but swaps weighted CE for
``FocalLoss(gamma=2)``. Early stopping is now on val frustration recall.

Outputs
-------
* ``checkpoints/roberta_text_v5.pt``
* ``data/processed/roberta_predictions_{train,val,test}_v5.csv``
* ``data/processed/dayG1_summary.json``

Stop condition (per the plan): if after 5 epochs val frust_recall <= 0.32
the run aborts and we fall back to v4 RoBERTa predictions for Phase 2.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.training.train_text import (  # noqa: E402
    TARGET_LABELS,
    TextOnlyClassifier,
    evaluate_on_test,
    load_meld_texts,
    load_synthetic_texts,
    split_synthetic,
)
from src.training.train_text_v5 import train_model_v5  # noqa: E402

DATA_DIR = os.path.join("data", "processed")
CKPT_DIR = "checkpoints"
SYNTH_LOCAL = os.path.join("data", "synthetic", "text_filtered.jsonl")
SYNTH_MAIN_WORKTREE = os.environ.get("SYNTH_TEXT_PATH", "")


def _resolve_synth_path(override: str | None) -> str:
    candidates = [c for c in (override, SYNTH_LOCAL, SYNTH_MAIN_WORKTREE) if c]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "Synthetic text JSONL not found. Tried: " + ", ".join(candidates),
    )


def _save_predictions(model, eval_data, split_name: str, device: str) -> Dict:
    df = evaluate_on_test(model, device=device, test_data=eval_data)
    out_path = os.path.join(DATA_DIR, f"roberta_predictions_{split_name}_v5.csv")
    df.to_csv(out_path, index=False)

    label_to_id = {l: i for i, l in enumerate(TARGET_LABELS)}
    y_true = df["true_label"].map(label_to_id).to_numpy()
    y_pred = df["predicted_class"].map(label_to_id).to_numpy()
    m = compute_metrics(y_true, y_pred, target_names=TARGET_LABELS)
    summary = {
        "n":            int(len(df)),
        "weighted_f1":  float(m["weighted_f1"]),
        "macro_f1":     float(m["macro_f1"]),
        "frust_recall": float(m["frustration_recall"]),
        "frust_f1":     float(m["per_class"].get("frustration", {}).get("f1", 0.0)),
        "anger_recall": float(m["per_class"].get("anger", {}).get("recall", 0.0)),
        "csv":          out_path,
    }
    print(f"  {split_name}: W-F1={summary['weighted_f1']:.4f} "
          f"M-F1={summary['macro_f1']:.4f} FrustR={summary['frust_recall']:.4f} "
          f"-> {out_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Day G1: focal-loss RoBERTa v5.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--synth-path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-abort-epoch", type=int, default=5,
                        help="If val frust_recall<=threshold by this epoch, abort.")
    parser.add_argument("--early-abort-threshold", type=float, default=0.32)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    t_start = time.time()

    print("=" * 78)
    print("  Day G1 - v5 RoBERTa fine-tune with FocalLoss(gamma=2)")
    print("=" * 78)

    synth_path = _resolve_synth_path(args.synth_path)
    print(f"[Data] synth source: {synth_path}")

    print("[Data] Loading MELD train ...", flush=True)
    meld_train = load_meld_texts("train")
    print(f"       {len(meld_train[0])} samples")
    print("[Data] Loading MELD val ...", flush=True)
    meld_val = load_meld_texts("validation")
    print(f"       {len(meld_val[0])} samples")
    print("[Data] Loading MELD test ...", flush=True)
    meld_test = load_meld_texts("test")
    print(f"       {len(meld_test[0])} samples")

    print("[Data] Loading synthetic ...", flush=True)
    synth_texts, synth_labels = load_synthetic_texts(synth_path)
    print(f"       {len(synth_texts)} samples")
    synth = split_synthetic(synth_texts, synth_labels, val_frac=0.1,
                              test_frac=0.1, seed=args.seed)

    combined_train: Tuple[List[str], List[int]] = (
        list(meld_train[0]) + list(synth["train"][0]),
        list(meld_train[1]) + list(synth["train"][1]),
    )
    print(f"       combined train: {len(combined_train[0])} samples "
          f"(MELD={len(meld_train[0])}, synth_train={len(synth['train'][0])})")

    print("\n[Train] focal loss")
    model, history = train_model_v5(
        train_data=combined_train,
        val_data=meld_val,
        max_epochs=args.epochs,
        patience=args.patience,
        gamma=args.gamma,
        batch_size=args.batch_size,
        checkpoint_dir=CKPT_DIR,
        checkpoint_name="roberta_text_v5.pt",
    )
    train_secs = time.time() - t_start

    aborted = False
    abort_reason = None
    epochs_run = max(history.get("epoch", [0])) if history.get("epoch") else 0
    if (epochs_run >= args.early_abort_epoch
            and history["best_val_frust_recall"] <= args.early_abort_threshold):
        aborted = True
        abort_reason = (
            f"focal loss did not improve over weighted CE in this configuration: "
            f"best val frust_recall={history['best_val_frust_recall']:.4f} "
            f"<= {args.early_abort_threshold:.2f} after {epochs_run} epochs."
        )
        print(f"\n[ABORT] {abort_reason}")

    print("\n[Eval] Generating MELD predictions on train/val/test ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_summaries = {
        "train": _save_predictions(model, meld_train, "train", device),
        "val":   _save_predictions(model, meld_val,   "val",   device),
        "test":  _save_predictions(model, meld_test,  "test",  device),
    }

    history_for_json = {k: v for k, v in history.items()
                          if k in ("epoch", "train_loss", "val_loss",
                                    "val_wf1", "val_mf1", "val_frust_recall")}

    summary = {
        "config": {
            "loss":       "FocalLoss",
            "gamma":      float(args.gamma),
            "use_class_weights": True,
            "lr":         2e-5,
            "freeze_epochs": 2,
            "unfreeze_top_n": 4,
            "max_epochs": args.epochs,
            "patience":   args.patience,
            "batch_size": args.batch_size,
            "synth_path": synth_path,
            "seed":       args.seed,
        },
        "training_time_sec":   round(train_secs, 1),
        "best_epoch":          int(history["best_epoch"]),
        "best_val_frust_recall": float(history["best_val_frust_recall"]),
        "best_val_w_f1":       float(history["val_wf1"][history["best_epoch"] - 1])
                                if history.get("val_wf1") else None,
        "best_val_macro_f1":   float(history["val_mf1"][history["best_epoch"] - 1])
                                if history.get("val_mf1") else None,
        "test_w_f1":           eval_summaries["test"]["weighted_f1"],
        "test_macro_f1":       eval_summaries["test"]["macro_f1"],
        "test_frust_recall":   eval_summaries["test"]["frust_recall"],
        "test_frust_f1":       eval_summaries["test"]["frust_f1"],
        "history":             history_for_json,
        "eval_per_split":      eval_summaries,
        "aborted":             aborted,
        "abort_reason":        abort_reason,
        "v4_reference":        {
            "comment": "v4 RoBERTa (combined_cw) was trained with weighted CE;"
                        " values below are from data/processed/roberta_predictions_v2.csv",
            "val_frust_recall":  0.15,
            "test_w_f1":         0.6537,
            "test_macro_f1":     0.4873,
            "test_frust_recall": 0.18,
        },
    }
    summary_path = os.path.join(DATA_DIR, "dayG1_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Done] summary -> {summary_path}")
    print(f"  total time: {train_secs/60:.1f} min")
    print(f"  best val frust_recall: {summary['best_val_frust_recall']:.4f}")
    print(f"  test:  W-F1={summary['test_w_f1']:.4f} "
          f"M-F1={summary['test_macro_f1']:.4f} "
          f"FrustR={summary['test_frust_recall']:.4f}")


if __name__ == "__main__":
    main()
