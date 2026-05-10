#!/usr/bin/env python3
"""SmartHandover - Day F15: Ensemble v3 (Phase 10).

Re-tunes the ensemble + handover threshold using **both** v2 components:

  * Text:  ``checkpoints/roberta_combined.pt``  (Day F8)
  * Audio: ``checkpoints/wav2vec2_finetuned.pt`` (Day F13, with-synth)

Pipeline
--------
1. Predict wav2vec2 fine-tuned over MELD train/val/test audio (~10-15
   min on the RTX 5060 Ti, leverages the cached .wav from Day F13).
2. Project the 5-class output back to the 4-dim "sb_*" schema used by
   the meta-classifier so the existing 19-dim feature layout stays
   intact:
      sb_ang = p_anger + p_frust          (consolidated negative-energy)
      sb_hap = p_satis
      sb_sad = p_sad
      sb_neu = p_neut
   This preserves backwards compatibility with ensemble_features_v2
   (we only overwrite the 4 sb_ columns).
3. Save audio_v3_predictions.csv (drop-in for speechbrain_predictions.csv).
4. Build features_v3 by replacing sb_* columns in features_v2.
5. Train meta_classifier_v3 (LR / XGB / MLP, pick best on val W-F1).
6. Run fusion_strategies on v3 features.
7. Sweep handover threshold on v3 meta-classifier output.
8. Print a summary table comparing v1 -> v2 -> v3.

Outputs (all under data/processed/):
* audio_v3_predictions.csv
* ensemble_features_{train,val,test}_v3.csv
* meta_classifier_v3.pkl + meta_classifier_v3_summary.json
* fusion_comparison_v3.{csv,png} + fusion_late_weights_v3.json
* dayF15_threshold_sweep.csv + configs/handover_threshold_v3.json
* dayF15_summary.json (the consolidated final-results blob)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classifiers.ensemble_trainer import (  # noqa: E402
    DATA_DIR, CKPT_DIR, FEATURE_COLUMNS, SB_COLS,
    TARGET_LABEL2ID, TARGET_LABELS,
)
from src.evaluation.metrics import compute_metrics, print_metrics  # noqa: E402

DEFAULT_AUDIO_CKPT = os.path.join("checkpoints", "wav2vec2_finetuned.pt")
SUFFIX = "_v3"


# ---------------------------------------------------------------------------
# Step 1 + 2 + 3: predict wav2vec2 fine-tuned on MELD + project to sb_*
# ---------------------------------------------------------------------------


AUDIO_V3_CSV = os.path.join(DATA_DIR, "audio_v3_predictions.csv")


def step_predict_audio(checkpoint: str, force: bool = False) -> pd.DataFrame:
    """Run the fine-tuned wav2vec2 over MELD train/val/test, save the
    projected sb_* columns to ``audio_v3_predictions.csv``.

    The CSV schema mirrors ``speechbrain_predictions.csv`` so the
    feature builder can drop it in unchanged.
    """
    if os.path.exists(AUDIO_V3_CSV) and not force:
        print(f"  cached -> {AUDIO_V3_CSV}")
        return pd.read_csv(AUDIO_V3_CSV)

    from src.training.train_audio import (
        load_meld_audio_records, predict_records,
    )

    print(f"  predicting with {checkpoint} ...")
    rows: List[Dict] = []
    for split_name, split_arg in (("train", "train"),
                                   ("val", "validation"),
                                   ("test", "test")):
        recs = load_meld_audio_records(splits=(split_arg,))
        print(f"    {split_name} ({len(recs)} clips) ...")
        preds = predict_records(checkpoint, recs)
        for i, p in enumerate(preds):
            audio_id = f"{split_name}_{i:05d}"
            # Consolidate 5-class -> sb_* (4-class IEMOCAP-style):
            sb_ang = p["p_anger"] + p["p_frust"]
            sb_hap = p["p_satis"]
            sb_sad = p["p_sad"]
            sb_neu = p["p_neut"]
            rows.append({
                "audio_id":        audio_id,
                "text":            "",            # filled-in later from the original CSV
                "true_label":      p["true_label"],
                "predicted_class": p["predicted_class"],
                "sb_ang":          float(sb_ang),
                "sb_hap":          float(sb_hap),
                "sb_sad":          float(sb_sad),
                "sb_neu":          float(sb_neu),
                # Keep raw 5-class for traceability (not used downstream)
                "raw_anger": p["p_anger"], "raw_frust": p["p_frust"],
                "raw_sad":   p["p_sad"],   "raw_neut":  p["p_neut"],
                "raw_satis": p["p_satis"],
            })

    df = pd.DataFrame(rows)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(AUDIO_V3_CSV, index=False)
    print(f"  saved -> {AUDIO_V3_CSV}  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Step 4: rebuild ensemble_features_*_v3.csv from features_v2 + audio_v3
# ---------------------------------------------------------------------------


def step_build_features_v3(audio_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Replace the sb_* columns of features_v2 with the new audio_v3 ones."""
    out: Dict[str, pd.DataFrame] = {}

    # Build a lookup audio_id -> sb_* (the audio_v3 CSV is row-aligned with
    # the order of MELD splits, same as the original speechbrain CSV).
    # We rebuild by joining on audio_id which is present in both v2
    # features and audio_v3.
    audio_lookup = {row["audio_id"]: row for _, row in audio_df.iterrows()}

    for split in ("train", "val", "test"):
        path_v2 = os.path.join(DATA_DIR, f"ensemble_features_{split}_v2.csv")
        if not os.path.exists(path_v2):
            print(f"  [WARN] {path_v2} not found - run Day F9 first.")
            return {}
        df = pd.read_csv(path_v2).copy()

        # Overwrite sb_* columns
        new_sb = []
        for aid in df["audio_id"].astype(str):
            row = audio_lookup.get(aid)
            if row is None:
                # Should not happen; keep zeros and warn
                new_sb.append((0.0, 0.0, 0.0, 0.0))
                continue
            new_sb.append((row["sb_ang"], row["sb_hap"],
                            row["sb_sad"], row["sb_neu"]))
        sb_arr = np.array(new_sb, dtype=np.float32)
        df["sb_ang"] = sb_arr[:, 0]
        df["sb_hap"] = sb_arr[:, 1]
        df["sb_sad"] = sb_arr[:, 2]
        df["sb_neu"] = sb_arr[:, 3]

        out_path = os.path.join(DATA_DIR, f"ensemble_features_{split}{SUFFIX}.csv")
        df.to_csv(out_path, index=False)
        out[split] = df
        print(f"    {split:<5s} {len(df):>5d} rows -> {out_path}")
    return out


# ---------------------------------------------------------------------------
# Step 5: train meta v3
# ---------------------------------------------------------------------------


def step_train_meta_v3() -> Dict:
    """Train candidates and pick best on val W-F1. Saves checkpoints/
    meta_classifier_v3.pkl and meta_classifier_v3_summary.json.
    """
    cmd = [
        sys.executable, "-m", "src.classifiers.ensemble_trainer",
        "--output-suffix", SUFFIX,
    ]
    print("  >>>", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"meta v3 training exited with {rc}")
    summary_path = os.path.join(DATA_DIR,
                                  f"meta_classifier{SUFFIX}_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Step 6: fusion comparison v3
# ---------------------------------------------------------------------------


def step_fusion_v3() -> pd.DataFrame:
    cmd = [
        sys.executable, "-m", "src.classifiers.fusion_strategies",
        "--features-suffix", SUFFIX,
        "--output-suffix",   SUFFIX,
    ]
    print("  >>>", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"fusion v3 exited with {rc}")
    csv_path = os.path.join(DATA_DIR, f"fusion_comparison{SUFFIX}.csv")
    return pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()


# ---------------------------------------------------------------------------
# Step 7: handover threshold sweep on the v3 meta-classifier
# ---------------------------------------------------------------------------


def step_threshold_sweep_v3() -> Dict:
    """Reproduce the Day-8 threshold sweep but on the v3 meta-classifier
    + v3 features. Picks t in [0.30, 0.70] maximising frustration_recall
    s.t. handover_precision > 0.50.
    """
    val_df = pd.read_csv(os.path.join(DATA_DIR,
                                       f"ensemble_features_val{SUFFIX}.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR,
                                        f"ensemble_features_test{SUFFIX}.csv"))
    bundle = joblib.load(os.path.join(CKPT_DIR,
                                       f"meta_classifier{SUFFIX}.pkl"))
    model = bundle["model"]
    feat_cols = bundle.get("feature_columns", FEATURE_COLUMNS)

    X_val  = val_df[feat_cols].to_numpy(dtype=np.float32)
    X_test = test_df[feat_cols].to_numpy(dtype=np.float32)
    y_val  = val_df["true_label_id"].to_numpy()
    y_test = test_df["true_label_id"].to_numpy()

    val_probs  = model.predict_proba(X_val)
    test_probs = model.predict_proba(X_test)

    anger_id = TARGET_LABEL2ID["anger"]
    frust_id = TARGET_LABEL2ID["frustration"]

    val_p_handover  = val_probs[:, anger_id]  + val_probs[:, frust_id]
    test_p_handover = test_probs[:, anger_id] + test_probs[:, frust_id]

    rows = []
    for t in np.arange(0.30, 0.70 + 1e-9, 0.01):
        t = float(round(t, 4))
        y_true_bin = np.isin(y_val, [anger_id, frust_id]).astype(int)
        y_pred_bin = (val_p_handover > t).astype(int)
        tp = int(((y_pred_bin == 1) & (y_true_bin == 1)).sum())
        fp = int(((y_pred_bin == 1) & (y_true_bin == 0)).sum())
        fn = int(((y_pred_bin == 0) & (y_true_bin == 1)).sum())
        tn = int(((y_pred_bin == 0) & (y_true_bin == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        frust_mask = (y_val == frust_id)
        frust_rec = ((y_pred_bin[frust_mask] == 1).sum()
                     / max(frust_mask.sum(), 1))
        rows.append({
            "threshold":           t,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "handover_precision":  prec,
            "handover_recall":     rec,
            "handover_f1":         f1,
            "frustration_recall":  frust_rec,
        })
    sweep = pd.DataFrame(rows)
    sweep_path = os.path.join(DATA_DIR, "dayF15_threshold_sweep.csv")
    sweep.to_csv(sweep_path, index=False)

    # Pick best: max frust_recall under precision > 0.50
    eligible = sweep[sweep["handover_precision"] > 0.50]
    if not eligible.empty:
        best = eligible.sort_values(
            ["frustration_recall", "handover_recall", "threshold"],
            ascending=[False, False, True],
        ).iloc[0]
        rule = "max frust_recall s.t. precision > 0.50 (val)"
    else:
        best = sweep.sort_values("handover_f1", ascending=False).iloc[0]
        rule = ("fallback: no threshold met precision > 0.50; "
                "picked max handover_f1")

    # Verify on test set
    t = float(best["threshold"])
    y_true_bin = np.isin(y_test, [anger_id, frust_id]).astype(int)
    y_pred_bin = (test_p_handover > t).astype(int)
    tp = int(((y_pred_bin == 1) & (y_true_bin == 1)).sum())
    fp = int(((y_pred_bin == 1) & (y_true_bin == 0)).sum())
    fn = int(((y_pred_bin == 0) & (y_true_bin == 1)).sum())
    tn = int(((y_pred_bin == 0) & (y_true_bin == 0)).sum())
    test_prec = tp / (tp + fp) if (tp + fp) else 0.0
    test_rec  = tp / (tp + fn) if (tp + fn) else 0.0
    test_f1   = (2 * test_prec * test_rec / (test_prec + test_rec)
                 ) if (test_prec + test_rec) else 0.0
    frust_mask = (y_test == frust_id)
    test_frust_rec = ((y_pred_bin[frust_mask] == 1).sum()
                      / max(frust_mask.sum(), 1))

    config = {
        "optimal_threshold":       t,
        "selection_rule":          rule,
        "validation_metrics": {
            "handover_precision": float(best["handover_precision"]),
            "handover_recall":    float(best["handover_recall"]),
            "handover_f1":        float(best["handover_f1"]),
            "frustration_recall": float(best["frustration_recall"]),
        },
        "test_metrics": {
            "handover_precision": test_prec,
            "handover_recall":    test_rec,
            "handover_f1":        test_f1,
            "frustration_recall": float(test_frust_rec),
        },
    }
    out_path = os.path.join("configs", f"handover_threshold{SUFFIX}.json")
    os.makedirs("configs", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"  threshold sweep -> {sweep_path}")
    print(f"  threshold config -> {out_path}")
    return config


# ---------------------------------------------------------------------------
# Step 8: print final comparison table
# ---------------------------------------------------------------------------


def _read_fusion_csv(suffix: str) -> Dict[str, Dict]:
    path = os.path.join(DATA_DIR, f"fusion_comparison{suffix}.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return {row["strategy"]: dict(row) for _, row in df.iterrows()}


def step_print_summary() -> None:
    print()
    print("=" * 80)
    print("  Day F15 - Final ensemble comparison (v1 -> v2 -> v3)")
    print("=" * 80)
    v1 = _read_fusion_csv("")
    v2 = _read_fusion_csv("_v2")
    v3 = _read_fusion_csv(SUFFIX)
    metrics_to_show = ("weighted_f1", "macro_f1", "frust_recall")

    strategies = list(v1.keys()) or list(v2.keys()) or list(v3.keys())
    for metric in metrics_to_show:
        print(f"\n  {metric}")
        print(f"  {'strategy':<24s} {'v1':>9s} {'v2':>9s} {'v3':>9s} "
              f"{'Δ (v3-v1)':>11s}")
        print("  " + "-" * 65)
        for s in strategies:
            v1v = v1.get(s, {}).get(metric, float("nan"))
            v2v = v2.get(s, {}).get(metric, float("nan"))
            v3v = v3.get(s, {}).get(metric, float("nan"))
            delta = (v3v - v1v) if (not np.isnan(v3v) and not np.isnan(v1v)) else float("nan")
            print(f"  {s:<24s} "
                  f"{v1v:>9.4f} {v2v:>9.4f} {v3v:>9.4f} "
                  f"{delta:>+11.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio-checkpoint", default=DEFAULT_AUDIO_CKPT)
    p.add_argument("--skip-predict", action="store_true",
                   help="Reuse audio_v3_predictions.csv if present.")
    args = p.parse_args()

    if not os.path.exists(args.audio_checkpoint):
        print(f"[ERROR] audio checkpoint not found: {args.audio_checkpoint}",
              file=sys.stderr)
        sys.exit(2)

    print("=" * 80)
    print("  Day F15 - Ensemble v3 (Phase 10)")
    print("=" * 80)
    print(f"  audio checkpoint : {args.audio_checkpoint}")

    print("\n[Step 1/4] Predict wav2vec2 fine-tuned over MELD audio ...")
    audio_df = step_predict_audio(args.audio_checkpoint,
                                    force=not args.skip_predict)

    print("\n[Step 2/4] Build features v3 (overwrite sb_* in features v2) ...")
    feat = step_build_features_v3(audio_df)
    if not feat:
        print("[ERROR] Could not build v3 features. Run Day F9 first to "
              "produce features_v2.")
        sys.exit(3)

    print("\n[Step 3/4] Train meta v3 + fusion v3 ...")
    step_train_meta_v3()
    step_fusion_v3()

    print("\n[Step 4/4] Handover threshold sweep on v3 ...")
    th_config = step_threshold_sweep_v3()
    print(f"  optimal threshold: {th_config['optimal_threshold']:.3f}")
    print(f"  test handover P/R/F1: "
          f"{th_config['test_metrics']['handover_precision']:.3f}/"
          f"{th_config['test_metrics']['handover_recall']:.3f}/"
          f"{th_config['test_metrics']['handover_f1']:.3f}")
    print(f"  test frust recall   : "
          f"{th_config['test_metrics']['frustration_recall']:.3f}")

    step_print_summary()

    # Persist consolidated summary
    summary = {
        "ensemble_version": "v3",
        "audio_checkpoint": args.audio_checkpoint,
        "threshold":        th_config,
    }
    with open(os.path.join(DATA_DIR, "dayF15_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Consolidated summary -> data/processed/dayF15_summary.json")


if __name__ == "__main__":
    main()
