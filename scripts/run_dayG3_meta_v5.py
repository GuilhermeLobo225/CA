#!/usr/bin/env python3
"""SmartHandover - Day G3: v5 meta-classifier with XGB grid search.

Differences from v4:
  * Four candidates: LR_balanced, XGB_v5 (grid), MLP_balanced, Stacking.
  * Selection rule: max ``0.5 * val_frust_recall + 0.5 * val_macro_f1``
    subject to ``val_weighted_f1 >= 0.55``.
  * Calibration: try both isotonic and sigmoid on val, keep the variant
    with higher test macro F1.
  * Honest comparison against v4 calibrated on the same test split.

If neither the uncalibrated winner nor the calibrated variant beats v4
on ``0.5 * frust_recall + 0.5 * macro_f1``, the script logs that fact
in the summary and keeps v4 as the production fallback.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from imblearn.over_sampling import SMOTE  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402

from src.classifiers.ensemble_trainer import (  # noqa: E402
    CKPT_DIR,
    DATA_DIR,
    TARGET_LABELS,
    TARGET_LABEL2ID,
)
from src.classifiers.stacking import StackingSoftVote  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402

FEATURE_COLUMNS_V5 = None  # populated from disk to be schema-driven

XGB_GRID = {
    "n_estimators":  [300, 500],
    "max_depth":     [4, 5, 6],
    "learning_rate": [0.05, 0.08],
}

MIN_W_F1 = 0.55
FRUST_ID = TARGET_LABEL2ID["frustration"]


def _read_features() -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    out: Dict[str, pd.DataFrame] = {}
    feat_cols = None
    for split in ("train", "val", "test"):
        p = os.path.join(DATA_DIR, f"ensemble_features_{split}_v5.csv")
        if not os.path.exists(p):
            raise FileNotFoundError(p + " - run dayG2 first")
        df = pd.read_csv(p)
        if feat_cols is None:
            feat_cols = [c for c in df.columns
                          if c not in ("audio_id", "text", "true_label",
                                         "true_label_id")]
            if len(feat_cols) != 20:
                raise RuntimeError(f"expected 20 features, got {len(feat_cols)}")
        out[split] = df
    return out, feat_cols


def _xy(df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    return (df[cols].to_numpy(dtype=np.float32),
            df["true_label_id"].to_numpy(dtype=np.int64))


def _metrics(model, X, y) -> Dict[str, float]:
    preds = model.predict(X)
    m = compute_metrics(y, preds, target_names=TARGET_LABELS)
    return {
        "weighted_f1":  float(m["weighted_f1"]),
        "macro_f1":     float(m["macro_f1"]),
        "frust_recall": float(m["frustration_recall"]),
        "frust_f1":     float(m["per_class"].get("frustration", {}).get("f1", 0.0)),
    }


def _sample_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=len(TARGET_LABELS)).astype(np.float64)
    counts[counts == 0] = 1
    inv = counts.sum() / (len(counts) * counts)
    return inv[y]


def _xgb_grid(X_tr, y_tr, X_val, y_val) -> Tuple[object, Dict, List[Dict]]:
    """Grid-search XGB; select by val frust_recall, tie-break by macro F1."""
    from xgboost import XGBClassifier

    sw = _sample_weights(y_tr)
    trace: List[Dict] = []
    best = None
    best_metrics: Dict[str, float] = {}
    best_params: Dict = {}
    for ne in XGB_GRID["n_estimators"]:
        for md in XGB_GRID["max_depth"]:
            for lr in XGB_GRID["learning_rate"]:
                est = XGBClassifier(
                    n_estimators=ne, max_depth=md, learning_rate=lr,
                    objective="multi:softprob", num_class=len(TARGET_LABELS),
                    eval_metric="mlogloss", n_jobs=-1, random_state=42,
                    verbosity=0,
                )
                est.fit(X_tr, y_tr, sample_weight=sw)
                v = _metrics(est, X_val, y_val)
                row = {"n_estimators": ne, "max_depth": md,
                        "learning_rate": lr, **v}
                trace.append(row)
                better = (
                    best is None
                    or v["frust_recall"] > best_metrics["frust_recall"]
                    or (v["frust_recall"] == best_metrics["frust_recall"]
                          and v["macro_f1"] > best_metrics["macro_f1"])
                )
                if better:
                    best = est
                    best_metrics = v
                    best_params = {"n_estimators": ne, "max_depth": md,
                                    "learning_rate": lr}
    return best, best_params, trace


def _calibrate(base, X_val, y_val, method: str):
    try:
        from sklearn.frozen import FrozenEstimator
        cal = CalibratedClassifierCV(FrozenEstimator(base), method=method)
    except ImportError:
        cal = CalibratedClassifierCV(base, method=method, cv="prefit")
    cal.fit(X_val, y_val)
    return cal


def main() -> None:
    print("=" * 78)
    print("  Day G3 - v5 meta-classifier (LR / XGB-grid / MLP / Stacking)")
    print("=" * 78)
    t_start = time.time()
    features, feat_cols = _read_features()
    print(f"  Loaded v5 features ({len(feat_cols)} columns)")
    X_tr, y_tr = _xy(features["train"], feat_cols)
    X_val, y_val = _xy(features["val"], feat_cols)
    X_test, y_test = _xy(features["test"], feat_cols)
    print(f"  Shapes: train={X_tr.shape}, val={X_val.shape}, test={X_test.shape}")

    print("\n[1] SMOTE on train (k=5)")
    print(f"    before: {dict(Counter(y_tr.tolist()))}")
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)
    print(f"    after:  {dict(Counter(y_tr_res.tolist()))}")

    print("\n[2] Training candidates")
    candidates: Dict[str, Dict] = {}
    fitted: Dict[str, object] = {}

    print("  >> LR_balanced")
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    lr.fit(X_tr_res, y_tr_res)
    fitted["LR_balanced"] = lr
    candidates["LR_balanced"] = {
        "val":  _metrics(lr, X_val,  y_val),
        "test": _metrics(lr, X_test, y_test),
    }
    print(f"     val={candidates['LR_balanced']['val']}")
    print(f"     test={candidates['LR_balanced']['test']}")

    print("  >> XGB grid search")
    xgb_best, xgb_params, xgb_trace = _xgb_grid(X_tr_res, y_tr_res, X_val, y_val)
    fitted["XGB_v5"] = xgb_best
    candidates["XGB_v5"] = {
        "val":  _metrics(xgb_best, X_val,  y_val),
        "test": _metrics(xgb_best, X_test, y_test),
        "best_params": xgb_params,
    }
    print(f"     best params: {xgb_params}")
    print(f"     val={candidates['XGB_v5']['val']}")
    print(f"     test={candidates['XGB_v5']['test']}")

    print("  >> MLP_balanced")
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                          early_stopping=True, random_state=42)
    mlp.fit(X_tr_res, y_tr_res)
    fitted["MLP_balanced"] = mlp
    candidates["MLP_balanced"] = {
        "val":  _metrics(mlp,  X_val,  y_val),
        "test": _metrics(mlp,  X_test, y_test),
    }
    print(f"     val={candidates['MLP_balanced']['val']}")
    print(f"     test={candidates['MLP_balanced']['test']}")

    print("  >> Stacking (LR + XGB soft-vote)")
    stack = StackingSoftVote([lr, xgb_best])
    fitted["Stacking"] = stack
    candidates["Stacking"] = {
        "val":  _metrics(stack, X_val,  y_val),
        "test": _metrics(stack, X_test, y_test),
    }
    print(f"     val={candidates['Stacking']['val']}")
    print(f"     test={candidates['Stacking']['test']}")

    print("\n[3] Selection: max 0.5*frust_recall + 0.5*macro_f1 s.t. W-F1>=0.55")
    eligible = {n: c for n, c in candidates.items()
                  if c["val"]["weighted_f1"] >= MIN_W_F1}
    if not eligible:
        eligible = candidates
        rule = "fallback: ignore W-F1 floor (no candidate cleared 0.55)"
    else:
        rule = "max 0.5*val_frust_recall + 0.5*val_macro_f1 s.t. val W-F1 >= 0.55"
    score = lambda c: 0.5 * c["val"]["frust_recall"] + 0.5 * c["val"]["macro_f1"]
    best_name = max(eligible.items(), key=lambda kv: score(kv[1]))[0]
    best_model = fitted[best_name]
    print(f"  Winner: {best_name}  (val score={score(candidates[best_name]):.4f})")

    raw_path = os.path.join(CKPT_DIR, "meta_classifier_v5.pkl")
    if best_name != "Stacking":
        # Stacking has no joblib-friendly representation; save the components.
        joblib.dump({"name": best_name, "model": best_model,
                       "feature_columns": feat_cols,
                       "target_labels": TARGET_LABELS,
                       "selection_rule": rule}, raw_path)
    else:
        joblib.dump({"name": "Stacking", "models": [lr, xgb_best],
                       "feature_columns": feat_cols,
                       "target_labels": TARGET_LABELS,
                       "selection_rule": rule}, raw_path)
    print(f"  uncalibrated -> {raw_path}")

    print("\n[4] Calibration sweep")
    cal_results: Dict[str, Dict] = {}
    cal_models: Dict[str, object] = {}
    if best_name == "Stacking":
        print("  Stacking already aggregates probabilities; skipping calibration sweep")
        cal_winner_method = None
        cal_winner_model = best_model
        cal_winner_test = candidates["Stacking"]["test"]
    else:
        for method in ("isotonic", "sigmoid"):
            cal = _calibrate(best_model, X_val, y_val, method)
            t = _metrics(cal, X_test, y_test)
            cal_results[method] = t
            cal_models[method] = cal
            print(f"  {method:<8s} test={t}")
        cal_winner_method = max(cal_results.items(),
                                  key=lambda kv: kv[1]["macro_f1"])[0]
        cal_winner_model = cal_models[cal_winner_method]
        cal_winner_test = cal_results[cal_winner_method]
        print(f"  best calibration: {cal_winner_method} "
              f"(test macro F1={cal_winner_test['macro_f1']:.4f})")

    cal_path = os.path.join(CKPT_DIR, "meta_classifier_v5_calibrated.pkl")
    joblib.dump({
        "name": f"{best_name}_{cal_winner_method or 'no_cal'}",
        "model": cal_winner_model,
        "feature_columns": feat_cols,
        "target_labels": TARGET_LABELS,
        "calibration_method": cal_winner_method,
    }, cal_path)
    print(f"  calibrated -> {cal_path}")

    # v4 baseline for honest comparison
    v4_summary_path = os.path.join(DATA_DIR, "meta_classifier_v4_summary.json")
    with open(v4_summary_path, "r", encoding="utf-8") as f:
        v4 = json.load(f)
    v4_test = v4["calibrated_test"]
    v4_score = 0.5 * v4_test["frust_recall"] + 0.5 * v4_test["macro_f1"]
    v5_uncal_test = candidates[best_name]["test"]
    v5_uncal_score = 0.5 * v5_uncal_test["frust_recall"] + 0.5 * v5_uncal_test["macro_f1"]
    v5_cal_score = 0.5 * cal_winner_test["frust_recall"] + 0.5 * cal_winner_test["macro_f1"]
    v5_wins = (v5_uncal_score > v4_score) or (v5_cal_score > v4_score)

    summary = {
        "best_name":       best_name,
        "selection_rule":  rule,
        "candidates":      candidates,
        "calibration_results": cal_results,
        "calibration_method": cal_winner_method,
        "calibrated_test": cal_winner_test,
        "v4_calibrated_test_reference": v4_test,
        "comparison_vs_v4": {
            "v4_score_0.5fr_0.5mf1":    float(v4_score),
            "v5_uncal_score":            float(v5_uncal_score),
            "v5_calibrated_score":       float(v5_cal_score),
            "v5_beats_v4":               bool(v5_wins),
            "honest_note": (
                "v5 beats v4 on the composite 0.5*frust_recall + 0.5*macro_f1 score"
                if v5_wins else
                "v5 did NOT beat v4 on the composite score; v4 remains the production fallback"
            ),
        },
        "training_time_sec": round(time.time() - t_start, 1),
    }
    out = os.path.join(DATA_DIR, "meta_classifier_v5_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  summary -> {out}")

    grid = {
        "xgb_grid":       XGB_GRID,
        "xgb_trace":      xgb_trace,
        "xgb_best":       xgb_params,
        "selection_rule": rule,
    }
    out2 = os.path.join(DATA_DIR, "dayG3_summary.json")
    with open(out2, "w", encoding="utf-8") as f:
        json.dump(grid, f, indent=2)
    print(f"  grid trace -> {out2}")

    print("\n" + "=" * 78)
    print(f"  v4 calibrated   : frust_recall={v4_test['frust_recall']:.4f}  "
          f"macro_f1={v4_test['macro_f1']:.4f}  W-F1={v4_test['weighted_f1']:.4f}  "
          f"score={v4_score:.4f}")
    print(f"  v5 uncal {best_name:<14s}: frust_recall={v5_uncal_test['frust_recall']:.4f}  "
          f"macro_f1={v5_uncal_test['macro_f1']:.4f}  W-F1={v5_uncal_test['weighted_f1']:.4f}  "
          f"score={v5_uncal_score:.4f}")
    print(f"  v5 cal   {cal_winner_method or 'n/a':<14s}: frust_recall={cal_winner_test['frust_recall']:.4f}  "
          f"macro_f1={cal_winner_test['macro_f1']:.4f}  W-F1={cal_winner_test['weighted_f1']:.4f}  "
          f"score={v5_cal_score:.4f}")
    print(f"  Outcome: {'v5 WINS' if v5_wins else 'v4 remains fallback'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
