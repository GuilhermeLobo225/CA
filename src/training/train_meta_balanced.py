"""
SmartHandover - Balanced Meta-Classifier (frustration-recall focused)

Day 6 trained the meta-classifier optimising for weighted F1, which - given
the savage class imbalance (268 frustration vs 4709 neutral on train) -
collapses frustration to ~14% recall. This module retrains the same family
of meta-classifiers with three combined remedies:

  1. SMOTE oversampling of the frustration class on the training split.
  2. ``class_weight='balanced'`` everywhere it is supported.
  3. Model selection on **frustration recall** (subject to a minimum
     weighted-F1 floor), instead of plain weighted F1.

It also runs an isotonic calibration on top of the chosen estimator so
the downstream handover threshold operates on better-calibrated scores.

The script is non-destructive: the classic ``meta_classifier.pkl`` from
Day 6 is left alone and the new bundle is written to
``checkpoints/meta_classifier_balanced.pkl``.

Run:

    python -m src.training.train_meta_balanced
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.classifiers.ensemble_trainer import (  # noqa: E402
    CKPT_DIR,
    DATA_DIR,
    FEATURE_COLUMNS,
    TARGET_LABELS,
    TARGET_LABEL2ID,
)
from src.evaluation.metrics import compute_metrics, print_metrics  # noqa: E402

FRUST_ID = TARGET_LABEL2ID["frustration"]

OUT_CKPT = os.path.join(CKPT_DIR, "meta_classifier_balanced.pkl")
OUT_MANIFEST = os.path.join(DATA_DIR, "meta_classifier_balanced_summary.json")

# Minimum weighted-F1 the new model is allowed to drop to in exchange for
# higher frustration recall. Day 6 baseline test W-F1 is 0.647.
MIN_WEIGHTED_F1 = 0.55


def _load_split(split: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(os.path.join(DATA_DIR, f"ensemble_features_{split}.csv"))
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df["true_label_id"].to_numpy(dtype=np.int64)
    return X, y


def _smote_resample(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE so all classes match the majority class (neutral)."""
    print("  Original class distribution:")
    for i, lbl in enumerate(TARGET_LABELS):
        n = int((y == i).sum())
        print(f"    {lbl:<14s} {n:>5d}")

    sm = SMOTE(random_state=seed, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X, y)

    print("  After SMOTE:")
    for i, lbl in enumerate(TARGET_LABELS):
        n = int((y_res == i).sum())
        print(f"    {lbl:<14s} {n:>5d}")

    return X_res, y_res


def _candidates() -> List[Tuple[str, object]]:
    """Meta-classifier variants. All support predict_proba."""
    cands: List[Tuple[str, object]] = [
        ("LR_balanced",
         LogisticRegression(max_iter=2000, class_weight="balanced",
                            solver="lbfgs", n_jobs=-1)),
        ("MLP_balanced",
         MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                       early_stopping=True, random_state=42)),
    ]
    try:
        from xgboost import XGBClassifier
        # XGBoost has no built-in class_weight - we feed sample_weight at fit.
        cands.insert(1, (
            "XGB_balanced",
            XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.08,
                objective="multi:softprob", num_class=len(TARGET_LABELS),
                eval_metric="mlogloss", n_jobs=-1, random_state=42,
                verbosity=0,
            ),
        ))
    except ImportError:
        pass
    return cands


def _sample_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency sample weights (used only for XGBoost)."""
    counts = np.bincount(y, minlength=len(TARGET_LABELS)).astype(np.float64)
    counts[counts == 0] = 1
    inv = counts.sum() / (len(counts) * counts)
    return inv[y]


def _evaluate(model, X, y) -> Dict[str, float]:
    preds = model.predict(X)
    m = compute_metrics(y, preds, target_names=TARGET_LABELS)
    return {
        "weighted_f1": m["weighted_f1"],
        "macro_f1": m["macro_f1"],
        "frust_recall": m["frustration_recall"],
        "frust_precision": m["per_class"].get("frustration", {}).get("precision", 0.0),
        "frust_f1": m["per_class"].get("frustration", {}).get("f1", 0.0),
    }


def main() -> None:
    print("=" * 72)
    print("  SmartHandover - Balanced Meta-Classifier (frustration-focused)")
    print("=" * 72)

    X_train, y_train = _load_split("train")
    X_val,   y_val   = _load_split("val")
    X_test,  y_test  = _load_split("test")
    print(f"\n  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    print("\n[1/3] SMOTE-resampling training set ...")
    X_tr_res, y_tr_res = _smote_resample(X_train, y_train)

    print("\n[2/3] Training candidate meta-classifiers ...")
    summary: Dict[str, Dict[str, float]] = {}
    fitted: Dict[str, object] = {}

    for name, est in _candidates():
        print(f"\n  >>> {name}")
        if name.startswith("XGB"):
            sw = _sample_weights(y_tr_res)
            est.fit(X_tr_res, y_tr_res, sample_weight=sw)
        else:
            est.fit(X_tr_res, y_tr_res)

        val_m = _evaluate(est, X_val, y_val)
        test_m = _evaluate(est, X_test, y_test)
        summary[name] = {
            "val":  val_m,
            "test": test_m,
        }
        fitted[name] = est

        print(f"    val  W-F1={val_m['weighted_f1']:.4f}  M-F1={val_m['macro_f1']:.4f}  "
              f"FrustR={val_m['frust_recall']:.4f}  FrustF1={val_m['frust_f1']:.4f}")
        print(f"    test W-F1={test_m['weighted_f1']:.4f}  M-F1={test_m['macro_f1']:.4f}  "
              f"FrustR={test_m['frust_recall']:.4f}  FrustF1={test_m['frust_f1']:.4f}")

    # --- Selection rule -----------------------------------------------------
    # Pick the candidate with highest validation frust_recall, subject to
    # validation weighted_f1 >= MIN_WEIGHTED_F1. If none qualify, pick the
    # one with the highest val frust_f1.
    eligible = [
        (name, vals) for name, vals in summary.items()
        if vals["val"]["weighted_f1"] >= MIN_WEIGHTED_F1
    ]
    if eligible:
        best_name, _ = max(eligible, key=lambda kv: kv[1]["val"]["frust_recall"])
        rule = (f"max val frust_recall s.t. val weighted_f1 >= {MIN_WEIGHTED_F1}")
    else:
        best_name, _ = max(summary.items(), key=lambda kv: kv[1]["val"]["frust_f1"])
        rule = "fallback: max val frust_f1 (no candidate met the W-F1 floor)"

    best_model = fitted[best_name]

    print("\n" + "=" * 72)
    print(f"  Selection rule : {rule}")
    print(f"  Chosen model   : {best_name}")
    print("=" * 72)

    print("\n[3/3] Isotonic calibration on validation set ...")
    # CalibratedClassifierCV(prefit=...) was deprecated in sklearn 1.4 in
    # favour of FrozenEstimator wrapping. Try both for compatibility.
    try:
        from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6
        calibrated = CalibratedClassifierCV(
            FrozenEstimator(best_model), method="isotonic"
        )
    except ImportError:
        calibrated = CalibratedClassifierCV(
            best_model, method="isotonic", cv="prefit"
        )
    calibrated.fit(X_val, y_val)

    # Final evaluation
    final_test = _evaluate(calibrated, X_test, y_test)
    print(f"  After calibration -> test W-F1={final_test['weighted_f1']:.4f}  "
          f"FrustR={final_test['frust_recall']:.4f}  FrustF1={final_test['frust_f1']:.4f}")

    test_preds = calibrated.predict(X_test)
    print()
    print_metrics(compute_metrics(y_test, test_preds, target_names=TARGET_LABELS))

    # --- Persist ------------------------------------------------------------
    bundle = {
        "name": f"{best_name}_smote_calibrated",
        "model": calibrated,
        "feature_columns": FEATURE_COLUMNS,
        "target_labels": TARGET_LABELS,
        "selection_rule": rule,
        "min_weighted_f1": MIN_WEIGHTED_F1,
    }
    os.makedirs(CKPT_DIR, exist_ok=True)
    joblib.dump(bundle, OUT_CKPT)
    print(f"\n  Saved -> {OUT_CKPT}")

    manifest = {
        "checkpoint": OUT_CKPT,
        "selection_rule": rule,
        "best_pre_calibration": best_name,
        "candidates": summary,
        "final_test_after_calibration": final_test,
        "min_weighted_f1": MIN_WEIGHTED_F1,
    }
    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest saved -> {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
