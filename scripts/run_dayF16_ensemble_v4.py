#!/usr/bin/env python3
"""SmartHandover - Day F16: Ensemble v4 (frustration-recall focused).

Last optimisation pass before the report. Builds on v2/v3 with three
levers that don't require any re-training:

  A. **20 features** instead of 19. The v3 schema collapses the 5
     wav2vec2-finetuned probs into 4 sb_* columns, losing the
     frust-vs-anger distinction. v4 keeps all 5 native classes:
        audio_anger, audio_frust, audio_sad, audio_neut, audio_satis
     -> richer feature signal for the meta-classifier.

  B. **Frustration-recall selection** with floor on W-F1. The v2/v3
     candidates were picked by val W-F1, which prefers MLP (22% frust
     recall) over LR (28% frust recall). v4 picks the candidate with
     the highest val frust_recall under W-F1 >= 0.55.

  C. **SMOTE + isotonic calibration**. Both already implemented in
     train_meta_balanced.py - we apply them to the v4 features.

  D. **Threshold tuning per-class**. The single P(anger)+P(frust) > t
     rule is replaced by a weighted combination grid-searched on val.

Output
------
* ``data/processed/ensemble_features_{train,val,test}_v4.csv`` (20 cols)
* ``checkpoints/meta_classifier_v4.pkl`` (uncalibrated best candidate)
* ``checkpoints/meta_classifier_v4_calibrated.pkl`` (isotonic calibrated)
* ``data/processed/meta_classifier_v4_summary.json``
* ``data/processed/fusion_comparison_v4.csv``
* ``data/processed/dayF16_threshold_sweep_v4.csv``
* ``configs/handover_threshold_v4.json``
* ``data/processed/dayF16_summary.json`` (consolidated v1->v4 comparison)

Run
---
    python scripts/run_dayF16_ensemble_v4.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from src.classifiers.ensemble_trainer import (  # noqa: E402
    DATA_DIR, CKPT_DIR, TARGET_LABELS, TARGET_LABEL2ID,
    GOEMO_COLS, ROBERTA_COLS, VADER_COLS,
)
from src.evaluation.metrics import compute_metrics, print_metrics  # noqa: E402

SUFFIX = "_v4"

# 20-dim schema: keep the original 5+6+4 text columns, replace 4 sb_*
# with 5 native audio columns from wav2vec2 fine-tuned.
AUDIO_COLS_V4 = ["audio_anger", "audio_frust", "audio_sad",
                  "audio_neut", "audio_satis"]
FEATURE_COLUMNS_V4 = ROBERTA_COLS + GOEMO_COLS + VADER_COLS + AUDIO_COLS_V4
assert len(FEATURE_COLUMNS_V4) == 20, "v4 schema must be 20-dim"

# Selection rule for v4 candidates
MIN_WEIGHTED_F1 = 0.55  # below this, candidate is rejected even if frust is high


# ===========================================================================
# Step 1: build v4 features
# ===========================================================================


def build_features_v4() -> Dict[str, pd.DataFrame]:
    """Replace sb_* in features_v2 with the 5 native audio columns from
    audio_v3_predictions.csv (which has both formats stored)."""

    audio_df = pd.read_csv(os.path.join(DATA_DIR, "audio_v3_predictions.csv"))
    # Build lookup audio_id -> 5 raw probs
    audio_lookup: Dict[str, Tuple[float, float, float, float, float]] = {}
    for _, row in audio_df.iterrows():
        audio_lookup[str(row["audio_id"])] = (
            float(row["raw_anger"]), float(row["raw_frust"]),
            float(row["raw_sad"]),   float(row["raw_neut"]),
            float(row["raw_satis"]),
        )

    out: Dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        v2 = pd.read_csv(os.path.join(
            DATA_DIR, f"ensemble_features_{split}_v2.csv"))
        df = v2.copy()

        # Drop sb_* (we won't use them in v4)
        for col in ("sb_ang", "sb_hap", "sb_sad", "sb_neu"):
            if col in df.columns:
                df = df.drop(columns=col)

        # Add audio_* (5 cols)
        new_audio = []
        for aid in df["audio_id"].astype(str):
            new_audio.append(audio_lookup.get(aid, (0., 0., 0., 0., 0.)))
        arr = np.array(new_audio, dtype=np.float32)
        df["audio_anger"] = arr[:, 0]
        df["audio_frust"] = arr[:, 1]
        df["audio_sad"]   = arr[:, 2]
        df["audio_neut"]  = arr[:, 3]
        df["audio_satis"] = arr[:, 4]

        # Reorder: meta cols + 20 features
        meta_cols = ["audio_id", "text", "true_label", "true_label_id"]
        df = df[meta_cols + FEATURE_COLUMNS_V4]

        out_path = os.path.join(DATA_DIR,
                                  f"ensemble_features_{split}{SUFFIX}.csv")
        df.to_csv(out_path, index=False)
        print(f"    {split:<5s}  {len(df):>5d} rows -> {out_path}")
        out[split] = df
    return out


# ===========================================================================
# Step 2: train candidates with class_weight + select on frust_recall
# ===========================================================================


def _xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = df[FEATURE_COLUMNS_V4].to_numpy(dtype=np.float32)
    y = df["true_label_id"].to_numpy(dtype=np.int64)
    return X, y


def _candidates() -> List[Tuple[str, object]]:
    cands = [
        ("LR_balanced",
         LogisticRegression(max_iter=2000, class_weight="balanced",
                              solver="lbfgs")),
        ("MLP_balanced",
         MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                         early_stopping=True, random_state=42)),
    ]
    try:
        from xgboost import XGBClassifier
        cands.insert(1, (
            "XGB_balanced",
            XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                            objective="multi:softprob",
                            num_class=len(TARGET_LABELS),
                            eval_metric="mlogloss", n_jobs=-1, random_state=42,
                            verbosity=0),
        ))
    except ImportError:
        pass
    return cands


def _sample_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=len(TARGET_LABELS)).astype(np.float64)
    counts[counts == 0] = 1
    inv = counts.sum() / (len(counts) * counts)
    return inv[y]


def _eval(model, X, y) -> Dict[str, float]:
    preds = model.predict(X)
    m = compute_metrics(y, preds, target_names=TARGET_LABELS)
    return {
        "weighted_f1":     m["weighted_f1"],
        "macro_f1":        m["macro_f1"],
        "frust_recall":    m["frustration_recall"],
        "frust_f1":        m["per_class"].get("frustration", {}).get("f1", 0.0),
    }


def step_train_meta_v4(features: Dict[str, pd.DataFrame]) -> Dict:
    X_train, y_train = _xy(features["train"])
    X_val,   y_val   = _xy(features["val"])
    X_test,  y_test  = _xy(features["test"])

    print(f"  Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    print()

    # SMOTE on training set
    print("  Applying SMOTE on training set ...")
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_tr_res, y_tr_res = sm.fit_resample(X_train, y_train)
    from collections import Counter
    print(f"    before: {dict(Counter(y_train))}")
    print(f"    after:  {dict(Counter(y_tr_res))}")
    print()

    summary: Dict[str, Dict] = {}
    fitted: Dict[str, object] = {}

    print("  Training candidates ...")
    for name, est in _candidates():
        print(f"\n  >>> {name}")
        if name.startswith("XGB"):
            sw = _sample_weights(y_tr_res)
            est.fit(X_tr_res, y_tr_res, sample_weight=sw)
        else:
            est.fit(X_tr_res, y_tr_res)
        v = _eval(est, X_val, y_val)
        t = _eval(est, X_test, y_test)
        summary[name] = {"val": v, "test": t}
        fitted[name] = est
        print(f"    val   W-F1={v['weighted_f1']:.4f}  M-F1={v['macro_f1']:.4f}  "
              f"FrustR={v['frust_recall']:.4f}")
        print(f"    test  W-F1={t['weighted_f1']:.4f}  M-F1={t['macro_f1']:.4f}  "
              f"FrustR={t['frust_recall']:.4f}")

    # Selection: max val frust_recall s.t. val W-F1 >= MIN_WEIGHTED_F1
    eligible = {n: v for n, v in summary.items()
                if v["val"]["weighted_f1"] >= MIN_WEIGHTED_F1}
    if eligible:
        best = max(eligible.items(),
                    key=lambda kv: kv[1]["val"]["frust_recall"])
        rule = f"max val frust_recall s.t. val W-F1 >= {MIN_WEIGHTED_F1}"
    else:
        best = max(summary.items(), key=lambda kv: kv[1]["val"]["frust_f1"])
        rule = "fallback: max val frust_f1"
    best_name = best[0]
    best_model = fitted[best_name]
    print(f"\n  >>> Selected: {best_name}  ({rule})")

    # Save uncalibrated
    raw_path = os.path.join(CKPT_DIR, f"meta_classifier{SUFFIX}.pkl")
    joblib.dump(
        {"name": best_name, "model": best_model,
         "feature_columns": FEATURE_COLUMNS_V4,
         "target_labels": TARGET_LABELS,
         "selection_rule": rule},
        raw_path,
    )
    print(f"  Saved -> {raw_path}")

    # Calibrate (isotonic on val set)
    print("\n  Applying isotonic calibration on val set ...")
    try:
        from sklearn.frozen import FrozenEstimator
        cal = CalibratedClassifierCV(FrozenEstimator(best_model),
                                       method="isotonic")
    except ImportError:
        cal = CalibratedClassifierCV(best_model, method="isotonic", cv="prefit")
    cal.fit(X_val, y_val)
    cal_test = _eval(cal, X_test, y_test)
    print(f"  After calibration -> test W-F1={cal_test['weighted_f1']:.4f}  "
          f"FrustR={cal_test['frust_recall']:.4f}")

    cal_path = os.path.join(CKPT_DIR, f"meta_classifier{SUFFIX}_calibrated.pkl")
    joblib.dump(
        {"name": f"{best_name}_calibrated", "model": cal,
         "feature_columns": FEATURE_COLUMNS_V4,
         "target_labels": TARGET_LABELS},
        cal_path,
    )
    print(f"  Saved -> {cal_path}")

    # Persist summary
    summary_path = os.path.join(DATA_DIR,
                                  f"meta_classifier{SUFFIX}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_name":      best_name,
            "selection_rule": rule,
            "candidates":     summary,
            "calibrated_test": cal_test,
        }, f, indent=2)
    print(f"  Summary -> {summary_path}")

    # Print final report
    print()
    print("=" * 60)
    print(f"  Final v4 test metrics ({best_name} + isotonic calibration)")
    print("=" * 60)
    test_preds = cal.predict(X_test)
    print_metrics(compute_metrics(y_test, test_preds,
                                    target_names=TARGET_LABELS))

    return {"best_name": best_name, "best_model": cal,
            "feature_columns": FEATURE_COLUMNS_V4,
            "candidates": summary,
            "calibrated_test": cal_test}


# ===========================================================================
# Step 3: threshold sweep with anger+frust binarisation
# ===========================================================================


def step_threshold_sweep(meta_bundle: Dict, features: Dict) -> Dict:
    model = meta_bundle["best_model"]
    val_df  = features["val"]
    test_df = features["test"]

    X_val  = val_df[FEATURE_COLUMNS_V4].to_numpy(dtype=np.float32)
    X_test = test_df[FEATURE_COLUMNS_V4].to_numpy(dtype=np.float32)
    y_val  = val_df["true_label_id"].to_numpy()
    y_test = test_df["true_label_id"].to_numpy()

    val_probs  = model.predict_proba(X_val)
    test_probs = model.predict_proba(X_test)

    a_id = TARGET_LABEL2ID["anger"]
    f_id = TARGET_LABEL2ID["frustration"]

    # Sweep over thresholds AND anger/frust weights
    rows = []
    for w_anger, w_frust in [(0.5, 0.5), (0.4, 0.6), (0.3, 0.7), (0.6, 0.4)]:
        val_score  = w_anger * val_probs[:, a_id]  + w_frust * val_probs[:, f_id]
        test_score = w_anger * test_probs[:, a_id] + w_frust * test_probs[:, f_id]
        for t in np.arange(0.20, 0.70 + 1e-9, 0.01):
            t = float(round(t, 4))
            for split, scores, y_true in (("val", val_score, y_val),
                                            ("test", test_score, y_test)):
                y_true_bin = np.isin(y_true, [a_id, f_id]).astype(int)
                y_pred_bin = (scores > t).astype(int)
                tp = int(((y_pred_bin == 1) & (y_true_bin == 1)).sum())
                fp = int(((y_pred_bin == 1) & (y_true_bin == 0)).sum())
                fn = int(((y_pred_bin == 0) & (y_true_bin == 1)).sum())
                tn = int(((y_pred_bin == 0) & (y_true_bin == 0)).sum())
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                rec  = tp / (tp + fn) if (tp + fn) else 0.0
                f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
                frust_mask = (y_true == f_id)
                frust_rec = ((y_pred_bin[frust_mask] == 1).sum()
                              / max(frust_mask.sum(), 1))
                rows.append({
                    "split":               split,
                    "w_anger":             w_anger,
                    "w_frust":             w_frust,
                    "threshold":           t,
                    "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                    "handover_precision":  prec,
                    "handover_recall":     rec,
                    "handover_f1":         f1,
                    "frustration_recall":  frust_rec,
                })
    sweep = pd.DataFrame(rows)
    sweep_path = os.path.join(DATA_DIR, "dayF16_threshold_sweep_v4.csv")
    sweep.to_csv(sweep_path, index=False)
    print(f"  threshold sweep -> {sweep_path}")

    # Pick best on val: max frust_recall s.t. precision > 0.50
    val_only = sweep[sweep["split"] == "val"].copy()
    eligible = val_only[val_only["handover_precision"] > 0.50]
    if not eligible.empty:
        best = eligible.sort_values(
            ["frustration_recall", "handover_recall", "threshold"],
            ascending=[False, False, True],
        ).iloc[0]
        rule = "max frust_recall s.t. precision > 0.50 (val)"
    else:
        best = val_only.sort_values("handover_f1", ascending=False).iloc[0]
        rule = "fallback: max handover_f1"

    # Apply same (w_anger, w_frust, t) to test
    test_match = sweep[
        (sweep["split"] == "test") &
        (sweep["w_anger"] == best["w_anger"]) &
        (sweep["w_frust"] == best["w_frust"]) &
        (sweep["threshold"] == best["threshold"])
    ].iloc[0]

    config = {
        "weights":           {"w_anger": float(best["w_anger"]),
                              "w_frust": float(best["w_frust"])},
        "optimal_threshold": float(best["threshold"]),
        "selection_rule":    rule,
        "validation_metrics": {k: float(best[k]) for k in
                                ["handover_precision", "handover_recall",
                                 "handover_f1", "frustration_recall"]},
        "test_metrics":      {k: float(test_match[k]) for k in
                                ["handover_precision", "handover_recall",
                                 "handover_f1", "frustration_recall"]},
    }
    out_path = os.path.join("configs", f"handover_threshold{SUFFIX}.json")
    os.makedirs("configs", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"  threshold config -> {out_path}")

    print(f"\n  Best on val: w_anger={best['w_anger']}, w_frust={best['w_frust']}, "
          f"t={best['threshold']:.3f}")
    print(f"    val P/R/F1: {best['handover_precision']:.3f} / "
          f"{best['handover_recall']:.3f} / {best['handover_f1']:.3f}")
    print(f"    val frust recall: {best['frustration_recall']:.3f}")
    print(f"   test P/R/F1: {test_match['handover_precision']:.3f} / "
          f"{test_match['handover_recall']:.3f} / "
          f"{test_match['handover_f1']:.3f}")
    print(f"   test frust recall: {test_match['frustration_recall']:.3f}")
    return config


# ===========================================================================
# Step 4: final v1->v4 comparison
# ===========================================================================


def step_print_summary(meta_v4: Dict, threshold_v4: Dict) -> None:
    print()
    print("=" * 80)
    print("  Day F16 - Final ensemble comparison (v1 -> v2 -> v3 -> v4)")
    print("=" * 80)

    # Headline meta-classifier metrics
    versions = [
        ("v1", "data/processed/meta_classifier_summary.json", "summary"),
        ("v2", "data/processed/meta_classifier_v2_summary.json", "summary"),
        ("v3", "data/processed/meta_classifier_v3_summary.json", "summary"),
    ]
    rows = []
    for vname, path, key in versions:
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            best = d.get("best_meta_classifier", "")
            tests = d[key][best] if best in d[key] else {}
            rows.append({"version": vname,
                         "best": best,
                         "test_W-F1":   tests.get("test_weighted_f1", float("nan")),
                         "test_M-F1":   tests.get("test_macro_f1", float("nan")),
                         "test_FrustR": tests.get("test_frust_recall", float("nan"))})
        except Exception:
            rows.append({"version": vname, "best": "?",
                         "test_W-F1": float("nan"),
                         "test_M-F1": float("nan"),
                         "test_FrustR": float("nan")})

    # v4 row
    rows.append({
        "version":     "v4 (calibrated)",
        "best":        meta_v4.get("best_name", ""),
        "test_W-F1":   meta_v4["calibrated_test"]["weighted_f1"],
        "test_M-F1":   meta_v4["calibrated_test"]["macro_f1"],
        "test_FrustR": meta_v4["calibrated_test"]["frust_recall"],
    })

    print(f"\n  Meta-classifier (test set):")
    print(f"  {'version':<18s} {'best':<22s} {'W-F1':>8s} {'M-F1':>8s} {'FrustR':>8s}")
    print("  " + "-" * 65)
    for r in rows:
        print(f"  {r['version']:<18s} {r['best']:<22s} "
              f"{r['test_W-F1']:>8.4f} {r['test_M-F1']:>8.4f} "
              f"{r['test_FrustR']:>8.4f}")

    # Handover threshold metrics (across versions)
    print(f"\n  Handover (P/R/F1, frust_recall on test set):")
    print(f"  {'version':<8s} {'thr':>6s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} "
          f"{'FrustR':>8s}")
    print("  " + "-" * 50)
    for vname, path in (("v1/v2", "configs/handover_threshold.json"),
                         ("v3", "configs/handover_threshold_v3.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                c = json.load(f)
            tm = c["test_metrics"]
            print(f"  {vname:<8s} {c['optimal_threshold']:>6.3f} "
                  f"{tm['handover_precision']:>7.3f} "
                  f"{tm['handover_recall']:>7.3f} "
                  f"{tm['handover_f1']:>7.3f} "
                  f"{tm['frustration_recall']:>8.3f}")
        except Exception:
            pass
    tm_v4 = threshold_v4["test_metrics"]
    print(f"  {'v4':<8s} {threshold_v4['optimal_threshold']:>6.3f} "
          f"{tm_v4['handover_precision']:>7.3f} "
          f"{tm_v4['handover_recall']:>7.3f} "
          f"{tm_v4['handover_f1']:>7.3f} "
          f"{tm_v4['frustration_recall']:>8.3f}  "
          f"<- weights anger={threshold_v4['weights']['w_anger']}, "
          f"frust={threshold_v4['weights']['w_frust']}")

    # Persist consolidated summary
    summary_path = os.path.join(DATA_DIR, "dayF16_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "ensemble_version": "v4",
            "meta_classifier_v4": {
                "best": meta_v4["best_name"],
                "calibrated_test": meta_v4["calibrated_test"],
            },
            "threshold_v4": threshold_v4,
            "feature_schema": FEATURE_COLUMNS_V4,
        }, f, indent=2, default=str)
    print(f"\n  Consolidated summary -> {summary_path}")


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    print("=" * 78)
    print("  Day F16 - Ensemble v4 (frustration-recall focused)")
    print("=" * 78)

    print("\n[Step 1/3] Building features v4 (20 dims, 5 native audio classes) ...")
    features = build_features_v4()

    print("\n[Step 2/3] Training meta v4 + calibration ...")
    meta_v4 = step_train_meta_v4(features)

    print("\n[Step 3/3] Threshold sweep v4 (per-weight grid + anger/frust binarisation) ...")
    threshold_v4 = step_threshold_sweep(meta_v4, features)

    step_print_summary(meta_v4, threshold_v4)


if __name__ == "__main__":
    main()
