"""
SmartHandover — Error Analysis & Threshold Optimisation (Day 8)

Two deliverables in one script:

1) **Error analysis.**  Runs the saved meta-classifier on the MELD test set
   and exports the Top-20 most damaging mistakes to
   ``data/processed/top_20_errors.csv``.  Errors are ranked by severity using
   a tiered scheme:

      Tier A (severity 3): False Negatives where true = frustration.
      Tier B (severity 2): False Positives where pred = frustration.
      Tier C (severity 1): High-confidence confusion between the negative
                           trio (anger <-> frustration <-> sadness) or a
                           strongly-confident neutral mis-call of a negative
                           utterance.

   Within each tier rows are sorted by model confidence (more confident
   errors are "worse" — the model was sure and wrong).

2) **Handover-threshold optimisation.**  Instead of plain argmax we define

        p_handover = P(anger) + P(frustration)

   and search a threshold t ∈ [0.30, 0.70] on the **validation** set that
   maximises Frustration Recall subject to Handover Precision > 0.50.
   Results and the chosen threshold are persisted to
   ``configs/handover_threshold.json`` (picked up by pipeline.py / handover.py).

Run:

    python -m src.evaluation.error_analysis
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

# Project imports -----------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.classifiers.ensemble_trainer import (  # noqa: E402
    CKPT_DIR,
    DATA_DIR,
    FEATURE_COLUMNS,
    TARGET_LABEL2ID,
    TARGET_LABELS,
)
from src.evaluation.metrics import compute_metrics, print_metrics  # noqa: E402

# Indices for the two emotions that trigger a handover
ANGER_ID = TARGET_LABEL2ID["anger"]
FRUSTRATION_ID = TARGET_LABEL2ID["frustration"]
SADNESS_ID = TARGET_LABEL2ID["sadness"]
NEUTRAL_ID = TARGET_LABEL2ID["neutral"]

NEGATIVE_IDS = {ANGER_ID, FRUSTRATION_ID, SADNESS_ID}

CONFIG_DIR = "configs"
THRESHOLD_JSON = os.path.join(CONFIG_DIR, "handover_threshold.json")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_meta_classifier(path: str = None):
    """Load the meta-classifier bundle saved by ensemble_trainer.py."""
    if path is None:
        path = os.path.join(CKPT_DIR, "meta_classifier.pkl")
    bundle = joblib.load(path)
    return bundle["model"], bundle


def load_feature_split(split: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"ensemble_features_{split}.csv")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------


def _tier_and_severity(row: pd.Series) -> Tuple[int, str]:
    """Assign a severity tier to an error row.

    Returns (severity_int, tier_label). Higher integer = worse.
    Rows that are not errors get severity 0.
    """
    t, p = int(row["true_label_id"]), int(row["predicted_class_id"])
    if t == p:
        return 0, "correct"

    # Tier A — false negative for frustration (most damaging)
    if t == FRUSTRATION_ID and p != FRUSTRATION_ID:
        return 3, "A_FN_frustration"

    # Tier B — false positive for frustration
    if t != FRUSTRATION_ID and p == FRUSTRATION_ID:
        return 2, "B_FP_frustration"

    # Tier C — severe confusion inside the negative trio or
    #          strong-neutral mis-call of a negative sample.
    if t in NEGATIVE_IDS and p in NEGATIVE_IDS:
        return 1, "C_confusion_negatives"
    if t in NEGATIVE_IDS and p == NEUTRAL_ID:
        return 1, "C_missed_negative_as_neutral"

    return 0, "other"  # ignored for the Top-20 shortlist


def build_error_report(
    test_df: pd.DataFrame, probs: np.ndarray, preds: np.ndarray,
) -> pd.DataFrame:
    """Attach prediction info + severity ranking to the test DataFrame."""
    n_classes = probs.shape[1]
    assert n_classes == len(TARGET_LABELS)

    out = test_df.copy()
    out["predicted_class_id"] = preds.astype(int)
    out["predicted_class"] = [TARGET_LABELS[i] for i in preds]
    out["confidence"] = probs.max(axis=1)

    for i, label in enumerate(TARGET_LABELS):
        out[f"meta_prob_{label}"] = probs[:, i]

    out["p_handover"] = probs[:, ANGER_ID] + probs[:, FRUSTRATION_ID]

    tiers = out.apply(_tier_and_severity, axis=1, result_type="expand")
    out["severity"] = tiers[0].astype(int)
    out["tier"] = tiers[1].astype(str)
    return out


def top_k_errors(err_df: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """Return the Top-K most damaging errors sorted by (severity, confidence)."""
    errors = err_df[err_df["severity"] > 0].copy()
    errors = errors.sort_values(
        by=["severity", "confidence"], ascending=[False, False]
    )
    return errors.head(k).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Handover-threshold sweep
# ---------------------------------------------------------------------------


def _handover_binary_metrics(
    y_true_ids: np.ndarray, p_handover: np.ndarray, threshold: float,
) -> Dict[str, float]:
    """Compute precision/recall/F1 for the binary handover decision.

    "Positive" (handover=True) corresponds to true label in {anger, frustration}.
    The model's decision is p_handover > threshold.
    """
    y_true_bin = np.isin(y_true_ids, [ANGER_ID, FRUSTRATION_ID]).astype(int)
    y_pred_bin = (p_handover > threshold).astype(int)

    tp = int(((y_pred_bin == 1) & (y_true_bin == 1)).sum())
    fp = int(((y_pred_bin == 1) & (y_true_bin == 0)).sum())
    fn = int(((y_pred_bin == 0) & (y_true_bin == 1)).sum())
    tn = int(((y_pred_bin == 0) & (y_true_bin == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Frustration-specific recall
    frust_mask = (y_true_ids == FRUSTRATION_ID)
    frust_recall = (
        (y_pred_bin[frust_mask] == 1).sum() / max(frust_mask.sum(), 1)
    )

    return {
        "threshold": float(threshold),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "handover_precision": float(precision),
        "handover_recall": float(recall),
        "handover_f1": float(f1),
        "frustration_recall": float(frust_recall),
    }


def sweep_thresholds(
    y_true_ids: np.ndarray, p_handover: np.ndarray,
    lo: float = 0.30, hi: float = 0.70, step: float = 0.01,
    min_precision: float = 0.50,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Grid-search a threshold.

    We maximise **frustration recall** subject to handover precision strictly
    greater than ``min_precision``.  If no threshold hits the precision floor
    we fall back to the one with the highest handover F1 and report it.

    Returns (sweep_df, best_record).
    """
    thresholds = np.round(np.arange(lo, hi + 1e-9, step), 4)
    rows = [_handover_binary_metrics(y_true_ids, p_handover, t) for t in thresholds]
    df = pd.DataFrame(rows)

    eligible = df[df["handover_precision"] > min_precision]
    if len(eligible):
        # Among eligible, pick the one with highest frustration recall, break
        # ties by handover recall, then by lowest threshold.
        eligible_sorted = eligible.sort_values(
            by=["frustration_recall", "handover_recall", "threshold"],
            ascending=[False, False, True],
        )
        best = eligible_sorted.iloc[0].to_dict()
        best["selection_rule"] = (
            f"max frustration_recall s.t. handover_precision > {min_precision}"
        )
    else:
        fallback = df.sort_values("handover_f1", ascending=False).iloc[0].to_dict()
        fallback["selection_rule"] = (
            f"fallback: no threshold met precision > {min_precision}, "
            "picked max handover_f1"
        )
        best = fallback

    return df, best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("  SmartHandover — Day 8: Error Analysis & Threshold Optimisation")
    print("=" * 60)

    model, bundle = load_meta_classifier()
    print(f"  Meta-classifier: {bundle.get('name', type(model).__name__)}")

    val_df = load_feature_split("val")
    test_df = load_feature_split("test")

    X_val = val_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_val = val_df["true_label_id"].to_numpy()

    X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_test = test_df["true_label_id"].to_numpy()

    # --- Probability outputs ----------------------------------------------
    if hasattr(model, "predict_proba"):
        val_probs = model.predict_proba(X_val)
        test_probs = model.predict_proba(X_test)
    else:
        # Defensive fallback — LR/MLP/XGB all provide predict_proba, but
        # if the user swaps in an exotic model we one-hot-encode the preds.
        val_preds_only = model.predict(X_val)
        test_preds_only = model.predict(X_test)
        val_probs = np.eye(len(TARGET_LABELS))[val_preds_only]
        test_probs = np.eye(len(TARGET_LABELS))[test_preds_only]

    val_preds = val_probs.argmax(axis=1)
    test_preds = test_probs.argmax(axis=1)

    # =====================================================================
    # 1) ERROR ANALYSIS
    # =====================================================================
    print("\n[1/2] Error analysis on MELD test set ...")
    print_metrics(compute_metrics(y_test, test_preds, target_names=TARGET_LABELS))

    err_df = build_error_report(test_df, test_probs, test_preds)
    top20 = top_k_errors(err_df, k=20)

    # Keep the most useful columns for manual inspection
    columns_to_export = [
        "audio_id", "text", "true_label", "predicted_class",
        "confidence", "p_handover",
        "severity", "tier",
    ] + [f"meta_prob_{l}" for l in TARGET_LABELS] + FEATURE_COLUMNS

    # `audio_id` is only present if features file includes it
    columns_to_export = [c for c in columns_to_export if c in top20.columns]

    top20_out = top20[columns_to_export]
    out_path = os.path.join(DATA_DIR, "top_20_errors.csv")
    top20_out.to_csv(out_path, index=False)
    print(f"\n  Top-20 errors saved -> {out_path}")

    # Console preview (sanitize text for cp1252-capable stdout)
    print("\n  Top-20 preview (tier | true -> pred | conf | text[:70])")
    for _, r in top20_out.iterrows():
        txt = str(r["text"])[:70].replace("\n", " ")
        safe_txt = txt.encode("ascii", errors="replace").decode("ascii")
        print(f"    {r['tier']:<28s} | "
              f"{r['true_label']:<12s} -> {r['predicted_class']:<12s} | "
              f"{r['confidence']:.2f} | {safe_txt}")

    tier_counts = (
        err_df[err_df["severity"] > 0]["tier"].value_counts().to_dict()
    )
    print("\n  Error tier distribution (test set):")
    for tier, cnt in sorted(tier_counts.items()):
        print(f"    {tier:<28s} {cnt}")

    # =====================================================================
    # 2) HANDOVER THRESHOLD OPTIMISATION
    # =====================================================================
    print("\n[2/2] Threshold sweep on VALIDATION set ...")
    val_p_handover = val_probs[:, ANGER_ID] + val_probs[:, FRUSTRATION_ID]
    sweep_df, best = sweep_thresholds(
        y_val, val_p_handover,
        lo=0.30, hi=0.70, step=0.01, min_precision=0.50,
    )

    print("\n  Top-10 thresholds (by frustration recall, val set):")
    top10 = sweep_df.sort_values(
        ["frustration_recall", "handover_recall"], ascending=[False, False]
    ).head(10)
    print(f"  {'thr':>5s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} "
          f"{'FrustR':>7s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'TN':>4s}")
    for _, r in top10.iterrows():
        print(f"  {r['threshold']:>5.2f} "
              f"{r['handover_precision']:>6.3f} "
              f"{r['handover_recall']:>6.3f} "
              f"{r['handover_f1']:>6.3f} "
              f"{r['frustration_recall']:>7.3f} "
              f"{int(r['TP']):>4d} {int(r['FP']):>4d} "
              f"{int(r['FN']):>4d} {int(r['TN']):>4d}")

    print("\n  >>> OPTIMAL THRESHOLD (validation) <<<")
    print(f"  rule                : {best['selection_rule']}")
    print(f"  threshold           : {best['threshold']:.3f}")
    print(f"  handover_precision  : {best['handover_precision']:.4f}")
    print(f"  handover_recall     : {best['handover_recall']:.4f}")
    print(f"  handover_f1         : {best['handover_f1']:.4f}")
    print(f"  frustration_recall  : {best['frustration_recall']:.4f}")

    # --- Verify on the TEST set with the chosen threshold ----------------
    test_p_handover = test_probs[:, ANGER_ID] + test_probs[:, FRUSTRATION_ID]
    test_bin = _handover_binary_metrics(y_test, test_p_handover, best["threshold"])
    print("\n  Test-set metrics at selected threshold "
          f"({best['threshold']:.3f}):")
    print(f"    handover_precision : {test_bin['handover_precision']:.4f}")
    print(f"    handover_recall    : {test_bin['handover_recall']:.4f}")
    print(f"    handover_f1        : {test_bin['handover_f1']:.4f}")
    print(f"    frustration_recall : {test_bin['frustration_recall']:.4f}")

    # --- Persist the threshold for downstream use ------------------------
    os.makedirs(CONFIG_DIR, exist_ok=True)
    config = {
        "optimal_threshold": float(best["threshold"]),
        "min_precision_constraint": 0.50,
        "search_range": [0.30, 0.70],
        "step": 0.01,
        "selection_rule": best["selection_rule"],
        "validation_metrics": {
            k: best[k] for k in
            ["handover_precision", "handover_recall",
             "handover_f1", "frustration_recall"]
        },
        "test_metrics": {
            k: test_bin[k] for k in
            ["handover_precision", "handover_recall",
             "handover_f1", "frustration_recall"]
        },
    }
    with open(THRESHOLD_JSON, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"\n  Threshold config saved -> {THRESHOLD_JSON}")

    # Also dump the full sweep for the report
    sweep_path = os.path.join(DATA_DIR, "threshold_sweep.csv")
    sweep_df.to_csv(sweep_path, index=False)
    print(f"  Full sweep saved       -> {sweep_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
