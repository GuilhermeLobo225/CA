#!/usr/bin/env python3
"""SmartHandover - Day G2: Build v5 ensemble feature CSVs.

Strategy
--------
* Read the v4 features (``ensemble_features_{train,val,test}_v4.csv``)
  and replace the five RoBERTa prob_* columns with v5 predictions if
  Phase 1 succeeded; otherwise keep the v4 columns and log the fallback.
* Keep the 15 other columns (goemo_*, vader_*, audio_*) untouched.
* Assert the schema is still 20-D plus 4 metadata columns.

Outputs
-------
* ``data/processed/ensemble_features_{train,val,test}_v5.csv``
* ``data/processed/dayG2_summary.json``
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.classifiers.ensemble_trainer import (  # noqa: E402
    DATA_DIR,
    GOEMO_COLS,
    ROBERTA_COLS,
    VADER_COLS,
)

AUDIO_COLS_V5 = ["audio_anger", "audio_frust", "audio_sad",
                  "audio_neut", "audio_satis"]
FEATURE_COLUMNS_V5 = ROBERTA_COLS + GOEMO_COLS + VADER_COLS + AUDIO_COLS_V5
META_COLS = ["audio_id", "text", "true_label", "true_label_id"]
assert len(FEATURE_COLUMNS_V5) == 20, "v5 schema must remain 20-dim"


def _load_v5_roberta() -> Dict[str, pd.DataFrame] | None:
    """Return {split: df with [text, true_label, prob_*]} if v5 preds exist."""
    out: Dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        path = os.path.join(DATA_DIR, f"roberta_predictions_{split}_v5.csv")
        if not os.path.exists(path):
            return None
        out[split] = pd.read_csv(path)
    return out


def _v5_success_per_dayg1() -> bool:
    """Check dayG1_summary.json to decide whether to use v5 predictions."""
    path = os.path.join(DATA_DIR, "dayG1_summary.json")
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        s = json.load(f)
    if s.get("aborted"):
        return False
    # Use v5 only if it beats v4 reference on val frust_recall, otherwise
    # any "improvement" is noise. The v4 RoBERTa val frust_recall was 0.15.
    v4_ref = s.get("v4_reference", {}).get("val_frust_recall", 0.15)
    return float(s.get("best_val_frust_recall", 0.0)) > v4_ref


def main() -> None:
    print("=" * 70)
    print("  Day G2 - v5 ensemble features")
    print("=" * 70)

    v5_preds = _load_v5_roberta()
    use_v5 = v5_preds is not None and _v5_success_per_dayg1()
    source = "v5" if use_v5 else "v4"
    print(f"  RoBERTa source: {source} "
          f"({'v5 preds available + Phase 1 cleared bar' if use_v5 else 'falling back to v4 prob_* columns'})")

    summaries: Dict[str, Dict] = {}
    for split in ("train", "val", "test"):
        v4_path = os.path.join(DATA_DIR, f"ensemble_features_{split}_v4.csv")
        df = pd.read_csv(v4_path)
        n0 = len(df)

        if use_v5:
            preds = v5_preds[split].copy()
            if len(preds) != len(df):
                raise RuntimeError(
                    f"Row mismatch in split={split}: v4={len(df)}, v5={len(preds)}",
                )
            # Order-sensitive: roberta_predictions_{split}_v5 was generated
            # from MELD load (same iteration order as v4 features built from
            # v2 features, which also iterates MELD).
            for col in ROBERTA_COLS:
                if col not in preds.columns:
                    raise RuntimeError(f"v5 preds missing column {col}")
                df[col] = preds[col].astype(np.float32).to_numpy()

        # Reorder columns deterministically
        df = df[META_COLS + FEATURE_COLUMNS_V5]
        assert df.shape[1] == 4 + 20, f"unexpected col count: {df.shape[1]}"

        out_path = os.path.join(DATA_DIR, f"ensemble_features_{split}_v5.csv")
        df.to_csv(out_path, index=False)
        print(f"  {split:<5s}  rows={n0}  -> {out_path}")
        summaries[split] = {
            "n":          int(n0),
            "out_path":   out_path,
        }

    summary = {
        "feature_schema":  FEATURE_COLUMNS_V5,
        "roberta_source":  source,
        "v5_used":         use_v5,
        "splits":          summaries,
        "block_sources": {
            "prob_*":   source,
            "goemo_*":  "v4 (unchanged from j-hartmann/emotion-english-distilroberta-base)",
            "vader_*":  "v4 (unchanged from VADER sentiment)",
            "audio_*":  "v4 (unchanged from wav2vec2_finetuned.pt)",
        },
    }
    out = os.path.join(DATA_DIR, "dayG2_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary -> {out}")


if __name__ == "__main__":
    main()
