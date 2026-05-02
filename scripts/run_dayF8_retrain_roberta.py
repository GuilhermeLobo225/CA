#!/usr/bin/env python3
"""SmartHandover - Day F8: Re-train RoBERTa with synthetic augmentation.

Trains the text-only RoBERTa classifier in up to 4 conditions and writes
a comparative table at the end. The MELD test split is the gold standard:
all conditions are evaluated against it. Cross-corpus eval (train MELD ->
test synth, train synth -> test MELD) is also reported.

Conditions
----------
1. ``meld_only``       MELD train (baseline). Equivalent to
                       checkpoints/roberta_text_only.pt; we re-train under
                       the same seed for a fair comparison.
2. ``synth_only``      Synthetic train only (cross-corpus baseline).
3. ``combined``        MELD train + synthetic train (no class weights -
                       lets the dataset balance speak for itself).
4. ``combined_cw``     Same as combined but with inverse-frequency class
                       weights in CrossEntropy.

Outputs
-------
* ``checkpoints/roberta_<condition>.pt`` per condition.
* ``data/processed/dayF8_results.csv`` summary table.
* ``data/processed/dayF8_<condition>_meld_test.csv`` per-condition
  predictions on MELD test (drives the metric in the table).
* ``data/processed/dayF8_<condition>_synth_test.csv`` predictions on
  the synthetic test set (cross-corpus check).

Usage
-----
    python scripts/run_dayF8_retrain_roberta.py                 # all 4
    python scripts/run_dayF8_retrain_roberta.py --condition combined
    python scripts/run_dayF8_retrain_roberta.py --epochs 5      # fast dry run
    python scripts/run_dayF8_retrain_roberta.py --skip-meld-only # skip baseline rerun
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src.evaluation.metrics import compute_metrics, print_metrics  # noqa: E402
from src.training.train_text import (  # noqa: E402
    TARGET_LABELS,
    TextOnlyClassifier,
    evaluate_on_test,
    load_meld_texts,
    load_synthetic_texts,
    split_synthetic,
    train_model,
)

CHECKPOINT_DIR = "checkpoints"
PROCESSED_DIR  = os.path.join("data", "processed")
SUMMARY_CSV    = os.path.join(PROCESSED_DIR, "dayF8_results.csv")
SYNTH_FILTERED = os.path.join("data", "synthetic", "text_filtered.jsonl")

CONDITIONS = ("meld_only", "synth_only", "combined", "combined_cw")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _summarise_predictions(df: pd.DataFrame, dataset_name: str) -> Dict:
    """Compute weighted/macro/per-class metrics on a predictions DataFrame."""
    label_to_id = {l: i for i, l in enumerate(TARGET_LABELS)}
    y_true = df["true_label"].map(label_to_id).to_numpy()
    y_pred = df["predicted_class"].map(label_to_id).to_numpy()
    m = compute_metrics(y_true, y_pred, target_names=TARGET_LABELS)
    return {
        "dataset":          dataset_name,
        "n":                len(df),
        "weighted_f1":      m["weighted_f1"],
        "macro_f1":         m["macro_f1"],
        "frust_recall":     m["frustration_recall"],
        "anger_recall":     m["per_class"].get("anger",        {}).get("recall", 0.0),
        "sad_recall":       m["per_class"].get("sadness",      {}).get("recall", 0.0),
        "neut_recall":      m["per_class"].get("neutral",      {}).get("recall", 0.0),
        "satis_recall":     m["per_class"].get("satisfaction", {}).get("recall", 0.0),
    }


def _train_one(
    condition: str,
    train_data: Tuple[List[str], List[int]],
    val_data:   Tuple[List[str], List[int]],
    use_class_weights: bool,
    epochs: int,
    patience: int,
) -> str:
    """Train one condition, return the path of the best checkpoint."""
    print()
    print("=" * 72)
    print(f"  Training condition: {condition}")
    print(f"  train_data: {len(train_data[0])}, val_data: {len(val_data[0])}, "
          f"class_weights: {use_class_weights}")
    print("=" * 72)
    ckpt_name = f"roberta_{condition}.pt"
    train_model(
        train_data=train_data,
        val_data=val_data,
        max_epochs=epochs,
        patience=patience,
        use_class_weights=use_class_weights,
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_name=ckpt_name,
    )
    return os.path.join(CHECKPOINT_DIR, ckpt_name)


def _evaluate_one(
    ckpt_path: str,
    eval_sets: Dict[str, Tuple[List[str], List[int]]],
    out_prefix: str,
) -> Dict[str, Dict]:
    """Load a checkpoint and evaluate on every (name -> (texts, labels)) set.

    Saves one CSV per evaluation set and returns a dict of summaries.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TextOnlyClassifier()
    model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                     weights_only=True))
    model.to(device)

    summaries: Dict[str, Dict] = {}
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    for name, data in eval_sets.items():
        df = evaluate_on_test(model, device=device, test_data=data)
        out_csv = os.path.join(PROCESSED_DIR, f"{out_prefix}_{name}.csv")
        df.to_csv(out_csv, index=False)
        summary = _summarise_predictions(df, dataset_name=name)
        summaries[name] = summary
        print(f"  -> {name}: W-F1={summary['weighted_f1']:.4f}, "
              f"Frust-R={summary['frust_recall']:.4f}, "
              f"Macro-F1={summary['macro_f1']:.4f}  ({out_csv})")
    return summaries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Day F8: re-train RoBERTa with synthetic augmentation.",
    )
    parser.add_argument("--condition", choices=CONDITIONS, default=None,
                        help="Run only one condition (default: all).")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Max epochs per condition (default 20). "
                             "Use a small value (e.g. 2) for dry runs.")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early-stopping patience on val W-F1.")
    parser.add_argument("--synth-path", default=SYNTH_FILTERED,
                        help="Path to filtered synthetic JSONL.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the synthetic train/val/test split.")
    parser.add_argument("--skip-meld-only", action="store_true",
                        help="Skip the meld_only condition (use existing "
                             "checkpoints/roberta_text_only.pt instead).")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Train only, don't evaluate (useful for resume).")
    args = parser.parse_args()

    # ---- Load all sources -------------------------------------------------
    print("=" * 72)
    print("  SmartHandover - Day F8: Re-train RoBERTa")
    print("=" * 72)
    print("\n[1/3] Loading sources ...")

    print("  MELD train ...", end=" ", flush=True)
    meld_train = load_meld_texts("train")
    print(f"{len(meld_train[0])} samples")

    print("  MELD val   ...", end=" ", flush=True)
    meld_val = load_meld_texts("validation")
    print(f"{len(meld_val[0])} samples")

    print("  MELD test  ...", end=" ", flush=True)
    meld_test = load_meld_texts("test")
    print(f"{len(meld_test[0])} samples")

    if not os.path.exists(args.synth_path):
        print(f"\n[ERROR] Synthetic file not found: {args.synth_path}")
        print("        Run filter_text.py first.")
        sys.exit(1)

    print(f"  Synthetic  ...", end=" ", flush=True)
    synth_texts, synth_labels = load_synthetic_texts(args.synth_path)
    print(f"{len(synth_texts)} samples (from {args.synth_path})")

    # 80/10/10 stratified split of the synthetic data
    print("  Stratified split synth (80/10/10) ...", end=" ", flush=True)
    synth = split_synthetic(synth_texts, synth_labels, val_frac=0.1,
                            test_frac=0.1, seed=args.seed)
    print(f"train={len(synth['train'][0])}, val={len(synth['val'][0])}, "
          f"test={len(synth['test'][0])}")

    # ---- Build the datasets per condition --------------------------------
    combined_train = (
        list(meld_train[0]) + list(synth["train"][0]),
        list(meld_train[1]) + list(synth["train"][1]),
    )

    condition_specs = {
        "meld_only": {
            "train": meld_train, "val": meld_val,
            "use_class_weights": True,
        },
        "synth_only": {
            "train": synth["train"], "val": synth["val"],
            "use_class_weights": True,
        },
        "combined": {
            "train": combined_train, "val": meld_val,
            "use_class_weights": False,
        },
        "combined_cw": {
            "train": combined_train, "val": meld_val,
            "use_class_weights": True,
        },
    }

    if args.condition is not None:
        conditions_to_run = [args.condition]
    else:
        conditions_to_run = list(CONDITIONS)
    if args.skip_meld_only and "meld_only" in conditions_to_run:
        conditions_to_run.remove("meld_only")

    # ---- Train each condition --------------------------------------------
    print("\n[2/3] Training ...")
    ckpt_paths: Dict[str, str] = {}
    for cond in conditions_to_run:
        spec = condition_specs[cond]
        ckpt_paths[cond] = _train_one(
            cond,
            train_data=spec["train"],
            val_data=spec["val"],
            use_class_weights=spec["use_class_weights"],
            epochs=args.epochs,
            patience=args.patience,
        )

    # If meld_only was skipped, fall back to the existing checkpoint
    if "meld_only" not in ckpt_paths:
        legacy = os.path.join(CHECKPOINT_DIR, "roberta_text_only.pt")
        if os.path.exists(legacy):
            ckpt_paths["meld_only"] = legacy
            print(f"  meld_only: using existing checkpoint {legacy}")

    # ---- Evaluate every checkpoint on MELD test + synth test -------------
    if args.skip_eval:
        print("\n[3/3] Skipping evaluation per --skip-eval.")
        return

    print("\n[3/3] Evaluating on MELD test (gold) and synth test (cross-corpus)")
    eval_sets = {
        "meld_test":  meld_test,
        "synth_test": synth["test"],
    }

    rows: List[Dict] = []
    for cond, ckpt in ckpt_paths.items():
        print(f"\n  >>> {cond}  ({ckpt})")
        summaries = _evaluate_one(ckpt, eval_sets, out_prefix=f"dayF8_{cond}")
        for ds_name, summary in summaries.items():
            rows.append({"condition": cond, **summary})

    df = pd.DataFrame(rows)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(SUMMARY_CSV, index=False)

    # ---- Pretty-print summary --------------------------------------------
    print()
    print("=" * 92)
    print("  Day F8 - Final summary")
    print("=" * 92)
    print(f"  {'condition':<14s} {'eval':<12s} {'n':>6s} "
          f"{'W-F1':>7s} {'M-F1':>7s} {'Frust-R':>8s} "
          f"{'Anger-R':>8s} {'Sad-R':>7s} {'Neut-R':>7s} {'Satis-R':>8s}")
    print("  " + "-" * 90)
    for r in rows:
        print(f"  {r['condition']:<14s} {r['dataset']:<12s} {r['n']:>6d} "
              f"{r['weighted_f1']:>7.4f} {r['macro_f1']:>7.4f} "
              f"{r['frust_recall']:>8.4f} {r['anger_recall']:>8.4f} "
              f"{r['sad_recall']:>7.4f} {r['neut_recall']:>7.4f} "
              f"{r['satis_recall']:>8.4f}")

    print(f"\n  CSV  -> {SUMMARY_CSV}")
    print(f"  Per-condition predictions in {PROCESSED_DIR}/dayF8_*_*.csv")

    # Save a small JSON manifest for the report
    manifest = {
        "conditions": list(ckpt_paths.keys()),
        "checkpoints": ckpt_paths,
        "synthetic_source": args.synth_path,
        "epochs": args.epochs,
        "seed": args.seed,
        "summary_csv": SUMMARY_CSV,
    }
    manifest_path = os.path.join(PROCESSED_DIR, "dayF8_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
