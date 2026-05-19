#!/usr/bin/env python3
"""SmartHandover - Day G4: Threshold tuning for v5 handover.

Two selection rules are persisted:
  * Rule A (v4 replica): max val frust_recall s.t. val handover_precision >= 0.50
  * Rule B (new):        max val handover_f1 (no precision floor)

The primary config (``configs/handover_threshold_v5.json``) uses Rule B.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.classifiers.ensemble_trainer import (  # noqa: E402
    DATA_DIR,
    CKPT_DIR,
    TARGET_LABELS,
    TARGET_LABEL2ID,
)
from src.classifiers.stacking import StackingSoftVote  # noqa: F401, E402

W_ANGER_GRID = [0.50, 0.55, 0.60, 0.65, 0.70]
THRESHOLD_GRID = list(np.round(np.linspace(0.10, 0.40, 16), 4))
A_ID = TARGET_LABEL2ID["anger"]
F_ID = TARGET_LABEL2ID["frustration"]


def _xy(df: pd.DataFrame, cols: List[str]):
    return (df[cols].to_numpy(dtype=np.float32),
            df["true_label_id"].to_numpy(dtype=np.int64))


def _compute(scores: np.ndarray, y: np.ndarray, t: float) -> Dict[str, float]:
    y_bin = np.isin(y, [A_ID, F_ID]).astype(int)
    y_hat = (scores > t).astype(int)
    tp = int(((y_hat == 1) & (y_bin == 1)).sum())
    fp = int(((y_hat == 1) & (y_bin == 0)).sum())
    fn = int(((y_hat == 0) & (y_bin == 1)).sum())
    tn = int(((y_hat == 0) & (y_bin == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    frust_mask = (y == F_ID)
    frust_rec = ((y_hat[frust_mask] == 1).sum() / max(frust_mask.sum(), 1))
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "handover_precision": float(prec),
        "handover_recall":    float(rec),
        "handover_f1":        float(f1),
        "frustration_recall": float(frust_rec),
    }


def main() -> None:
    print("=" * 78)
    print("  Day G4 - v5 threshold sweep")
    print("=" * 78)

    bundle = joblib.load(os.path.join(CKPT_DIR, "meta_classifier_v5_calibrated.pkl"))
    feat_cols = bundle["feature_columns"]
    model = bundle["model"]
    print(f"  Loaded {bundle['name']}, {len(feat_cols)} features")

    val_df = pd.read_csv(os.path.join(DATA_DIR, "ensemble_features_val_v5.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "ensemble_features_test_v5.csv"))
    X_val, y_val = _xy(val_df, feat_cols)
    X_test, y_test = _xy(test_df, feat_cols)

    val_probs = model.predict_proba(X_val)
    test_probs = model.predict_proba(X_test)

    rows = []
    for wa in W_ANGER_GRID:
        wf = round(1.0 - wa, 4)
        val_score = wa * val_probs[:, A_ID] + wf * val_probs[:, F_ID]
        test_score = wa * test_probs[:, A_ID] + wf * test_probs[:, F_ID]
        for t in THRESHOLD_GRID:
            t = float(t)
            v = _compute(val_score, y_val, t)
            te = _compute(test_score, y_test, t)
            for split, m in (("val", v), ("test", te)):
                rows.append({
                    "split": split, "w_anger": wa, "w_frust": wf, "threshold": t,
                    **m,
                })
    sweep = pd.DataFrame(rows)
    sweep_path = os.path.join(DATA_DIR, "dayG4_threshold_sweep_v5.csv")
    sweep.to_csv(sweep_path, index=False)
    print(f"  full sweep -> {sweep_path}")

    val_only = sweep[sweep["split"] == "val"].copy()

    # Rule A: max frust_recall s.t. handover_precision >= 0.50
    eligibleA = val_only[val_only["handover_precision"] >= 0.50]
    if not eligibleA.empty:
        bestA = eligibleA.sort_values(
            ["frustration_recall", "handover_recall", "threshold"],
            ascending=[False, False, True],
        ).iloc[0]
    else:
        bestA = val_only.sort_values("handover_f1", ascending=False).iloc[0]

    # Rule B: max handover_f1 (no floor)
    bestB = val_only.sort_values(
        ["handover_f1", "frustration_recall", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]

    def pick_test(b):
        return sweep[(sweep["split"] == "test") &
                     (sweep["w_anger"] == b["w_anger"]) &
                     (sweep["w_frust"] == b["w_frust"]) &
                     (sweep["threshold"] == b["threshold"])].iloc[0]
    testA = pick_test(bestA)
    testB = pick_test(bestB)

    def fmt(row, prefix=""):
        return {
            "weights": {"w_anger": float(row["w_anger"]),
                          "w_frust": float(row["w_frust"])},
            "optimal_threshold": float(row["threshold"]),
            f"{prefix}metrics": {
                "handover_precision":  float(row["handover_precision"]),
                "handover_recall":     float(row["handover_recall"]),
                "handover_f1":         float(row["handover_f1"]),
                "frustration_recall":  float(row["frustration_recall"]),
            },
        }

    config_primary = {
        "primary_rule":     "B (max val handover_f1, no precision floor)",
        "weights":          {"w_anger": float(bestB["w_anger"]),
                              "w_frust": float(bestB["w_frust"])},
        "optimal_threshold": float(bestB["threshold"]),
        "validation_metrics": {
            "handover_precision":  float(bestB["handover_precision"]),
            "handover_recall":     float(bestB["handover_recall"]),
            "handover_f1":         float(bestB["handover_f1"]),
            "frustration_recall":  float(bestB["frustration_recall"]),
        },
        "test_metrics": {
            "handover_precision":  float(testB["handover_precision"]),
            "handover_recall":     float(testB["handover_recall"]),
            "handover_f1":         float(testB["handover_f1"]),
            "frustration_recall":  float(testB["frustration_recall"]),
        },
        "rule_a_fallback": {
            "description":      "max val frust_recall s.t. val handover_precision >= 0.50",
            "weights":          {"w_anger": float(bestA["w_anger"]),
                                   "w_frust": float(bestA["w_frust"])},
            "optimal_threshold": float(bestA["threshold"]),
            "validation_metrics": {
                "handover_precision":  float(bestA["handover_precision"]),
                "handover_recall":     float(bestA["handover_recall"]),
                "handover_f1":         float(bestA["handover_f1"]),
                "frustration_recall":  float(bestA["frustration_recall"]),
            },
            "test_metrics": {
                "handover_precision":  float(testA["handover_precision"]),
                "handover_recall":     float(testA["handover_recall"]),
                "handover_f1":         float(testA["handover_f1"]),
                "frustration_recall":  float(testA["frustration_recall"]),
            },
        },
    }
    cfg_path = os.path.join("configs", "handover_threshold_v5.json")
    os.makedirs("configs", exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config_primary, f, indent=2)
    print(f"  config (primary=Rule B) -> {cfg_path}")

    print()
    print("  Rule A (max frust_R s.t. prec>=0.50):")
    print(f"    w_anger={bestA['w_anger']:.2f}, w_frust={bestA['w_frust']:.2f}, "
          f"t={bestA['threshold']:.4f}")
    print(f"    val   P={bestA['handover_precision']:.3f} R={bestA['handover_recall']:.3f} "
          f"F1={bestA['handover_f1']:.3f} FrustR={bestA['frustration_recall']:.3f}")
    print(f"    test  P={testA['handover_precision']:.3f} R={testA['handover_recall']:.3f} "
          f"F1={testA['handover_f1']:.3f} FrustR={testA['frustration_recall']:.3f}")
    print()
    print("  Rule B (max handover_F1):")
    print(f"    w_anger={bestB['w_anger']:.2f}, w_frust={bestB['w_frust']:.2f}, "
          f"t={bestB['threshold']:.4f}")
    print(f"    val   P={bestB['handover_precision']:.3f} R={bestB['handover_recall']:.3f} "
          f"F1={bestB['handover_f1']:.3f} FrustR={bestB['frustration_recall']:.3f}")
    print(f"    test  P={testB['handover_precision']:.3f} R={testB['handover_recall']:.3f} "
          f"F1={testB['handover_f1']:.3f} FrustR={testB['frustration_recall']:.3f}")


if __name__ == "__main__":
    main()
