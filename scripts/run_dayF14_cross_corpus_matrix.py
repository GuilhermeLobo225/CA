#!/usr/bin/env python3
"""SmartHandover - Day F14: Cross-corpus matrix (Phase 9).

Aggregates the results of Phase 1 (text fine-tune; ``dayF8_results.csv``)
and Phase 7 (audio fine-tune; ``dayF13_results.csv``) into a unified
cross-corpus picture - the central figure of the report.

Two matrices, side-by-side:

  * **Text** (rows = train condition, cols = test corpus)
        rows: meld_only, synth_only, combined
        cols: meld_test, synth_test
        (CREMA-D has only 12 fixed sentences - useless for text models.)

  * **Audio** (rows = train condition, cols = test corpus)
        rows: no-synth, with-synth, frozen (pure baseline from Day F12)
        cols: meld_test, cremad_test, synth_test

Both matrices report two metrics (W-F1 and frustration recall) and
are saved as PNG heatmaps + a unified CSV.

Inputs (all already on disk):
  * data/processed/dayF8_results.csv      (text)
  * data/processed/dayF13_results.csv     (audio fine-tuned)
  * data/processed/cremad_baseline_summary.json (audio frozen on CREMA-D)

Outputs:
  * data/processed/cross_corpus_matrix.csv
  * data/processed/cross_corpus_matrix.png
  * data/processed/cross_corpus_matrix_frust.png  (frust-recall variant)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROCESSED_DIR = os.path.join("data", "processed")
TEXT_RESULTS  = os.path.join(PROCESSED_DIR, "dayF8_results.csv")
AUDIO_RESULTS = os.path.join(PROCESSED_DIR, "dayF13_results.csv")
CREMAD_BASELINE = os.path.join(PROCESSED_DIR, "cremad_baseline_summary.json")

OUT_CSV       = os.path.join(PROCESSED_DIR, "cross_corpus_matrix.csv")
OUT_PNG_WF1   = os.path.join(PROCESSED_DIR, "cross_corpus_matrix.png")
OUT_PNG_FRUST = os.path.join(PROCESSED_DIR, "cross_corpus_matrix_frust.png")


# ---------------------------------------------------------------------------
# Build the two matrices
# ---------------------------------------------------------------------------


def _safe_float(x, default: float = float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def build_text_matrix(metric: str = "weighted_f1") -> pd.DataFrame:
    """Pivot dayF8 results into a (train_condition, test_corpus) table."""
    df = pd.read_csv(TEXT_RESULTS)
    # Drop combined_cw (Day F8 decision: not the chosen variant)
    df = df[df["condition"] != "combined_cw"].copy()
    # Rename for nicer presentation
    df["train"] = df["condition"].map({
        "meld_only":  "MELD only",
        "synth_only": "Synth only",
        "combined":   "MELD + Synth",
    })
    df["test"] = df["dataset"].map({
        "meld_test":  "MELD test",
        "synth_test": "Synth test",
    })
    pivot = df.pivot(index="train", columns="test", values=metric)
    pivot = pivot.reindex(["MELD only", "Synth only", "MELD + Synth"])
    pivot = pivot[["MELD test", "Synth test"]]
    return pivot


def build_audio_matrix(metric: str = "weighted_f1") -> pd.DataFrame:
    """Pivot dayF13 results + frozen baseline into a (train, test) table."""
    df = pd.read_csv(AUDIO_RESULTS)
    df["train"] = df["variant"].map({
        "no-synth":   "MELD + CREMA-D",
        "with-synth": "MELD + CREMA-D + Synth",
    })
    df["test"] = df["test_set"].map({
        "meld_test":   "MELD test",
        "cremad_test": "CREMA-D test",
        "synth_test":  "Synth test",
    })
    pivot = df.pivot(index="train", columns="test", values=metric)

    # Add the frozen wav2vec2-IEMOCAP baseline as a row.
    # (It was evaluated on CREMA-D in Day F12; for MELD/synth we don't
    # have a direct frozen run on those test splits, so the cells stay NaN.)
    frozen_row = {"MELD test": float("nan"),
                   "CREMA-D test": float("nan"),
                   "Synth test": float("nan")}
    if metric == "weighted_f1" and os.path.exists(CREMAD_BASELINE):
        with open(CREMAD_BASELINE, "r", encoding="utf-8") as f:
            data = json.load(f)
        frozen_row["CREMA-D test"] = _safe_float(data.get("weighted_f1"))

    pivot.loc["Frozen (IEMOCAP)"] = frozen_row
    pivot = pivot.reindex(["Frozen (IEMOCAP)",
                           "MELD + CREMA-D",
                           "MELD + CREMA-D + Synth"])
    pivot = pivot[["MELD test", "CREMA-D test", "Synth test"]]
    return pivot


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_matrices(text_mat: pd.DataFrame, audio_mat: pd.DataFrame,
                   out_path: str, title_metric: str = "Weighted F1") -> None:
    sns.set_style("white")
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5),
        gridspec_kw={"width_ratios": [text_mat.shape[1] + 0.5,
                                       audio_mat.shape[1] + 0.5]},
    )

    sns.heatmap(text_mat, annot=True, fmt=".3f",
                cmap="YlGnBu", vmin=0, vmax=1.0,
                cbar=True, ax=ax1, linewidths=0.5,
                annot_kws={"fontsize": 11, "fontweight": "bold"})
    ax1.set_title(f"Text - {title_metric}", fontsize=13, fontweight="bold")
    ax1.set_xlabel("test corpus", fontsize=11)
    ax1.set_ylabel("train condition", fontsize=11)
    ax1.tick_params(axis="x", rotation=20)
    ax1.tick_params(axis="y", rotation=0)

    sns.heatmap(audio_mat, annot=True, fmt=".3f",
                cmap="YlOrRd", vmin=0, vmax=1.0,
                cbar=True, ax=ax2, linewidths=0.5,
                annot_kws={"fontsize": 11, "fontweight": "bold"})
    ax2.set_title(f"Audio - {title_metric}", fontsize=13, fontweight="bold")
    ax2.set_xlabel("test corpus", fontsize=11)
    ax2.set_ylabel("train condition", fontsize=11)
    ax2.tick_params(axis="x", rotation=20)
    ax2.tick_params(axis="y", rotation=0)

    fig.suptitle("Cross-corpus generalisation - SmartHandover Phase 9",
                  fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pretty-print + persist
# ---------------------------------------------------------------------------


def to_unified_csv(text_wf1, text_fr, audio_wf1, audio_fr, path: str) -> None:
    rows: List[Dict] = []
    for matrix_name, wf1_df, fr_df in [
        ("text",  text_wf1,  text_fr),
        ("audio", audio_wf1, audio_fr),
    ]:
        for train in wf1_df.index:
            for test in wf1_df.columns:
                rows.append({
                    "modality":     matrix_name,
                    "train":        train,
                    "test":         test,
                    "weighted_f1":  wf1_df.loc[train, test],
                    "frust_recall": fr_df.loc[train, test]
                                       if test in fr_df.columns else float("nan"),
                })
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.4f")


def main() -> None:
    print("=" * 72)
    print("  Day F14 - Cross-corpus matrix")
    print("=" * 72)

    # Build matrices
    text_wf1   = build_text_matrix("weighted_f1")
    text_fr    = build_text_matrix("frust_recall")
    audio_wf1  = build_audio_matrix("weighted_f1")
    audio_fr   = build_audio_matrix("frust_recall")

    # Console preview
    print("\nText - Weighted F1\n" + "=" * 50)
    print(text_wf1.round(4).to_string())
    print("\nText - Frustration Recall\n" + "=" * 50)
    print(text_fr.round(4).to_string())
    print("\nAudio - Weighted F1\n" + "=" * 50)
    print(audio_wf1.round(4).to_string())
    print("\nAudio - Frustration Recall\n" + "=" * 50)
    print(audio_fr.round(4).to_string())

    # CSV
    to_unified_csv(text_wf1, text_fr, audio_wf1, audio_fr, OUT_CSV)
    print(f"\nUnified CSV saved -> {OUT_CSV}")

    # Heatmaps
    plot_matrices(text_wf1, audio_wf1, OUT_PNG_WF1, title_metric="Weighted F1")
    plot_matrices(text_fr,  audio_fr,  OUT_PNG_FRUST,
                   title_metric="Frustration Recall")
    print(f"Heatmap (W-F1)   -> {OUT_PNG_WF1}")
    print(f"Heatmap (FrustR) -> {OUT_PNG_FRUST}")

    # ------------------------------------------------------------------
    # Quick narrative reading for the report draft
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  Suggested report readings (deltas to highlight)")
    print("=" * 72)

    if "MELD only" in text_wf1.index and "MELD + Synth" in text_wf1.index:
        d_meld = (text_wf1.loc["MELD + Synth", "MELD test"]
                  - text_wf1.loc["MELD only",   "MELD test"])
        d_synth = (text_wf1.loc["MELD + Synth", "Synth test"]
                   - text_wf1.loc["MELD only",  "Synth test"])
        print(f"  Text: MELD+Synth vs MELD only -> "
              f"MELD test {d_meld:+.3f} W-F1 | "
              f"Synth test {d_synth:+.3f} W-F1")

    if ("MELD + CREMA-D" in audio_wf1.index
            and "MELD + CREMA-D + Synth" in audio_wf1.index):
        for col in audio_wf1.columns:
            v1 = audio_wf1.loc["MELD + CREMA-D", col]
            v2 = audio_wf1.loc["MELD + CREMA-D + Synth", col]
            if not (np.isnan(v1) or np.isnan(v2)):
                print(f"  Audio: + Synth in {col:<14s} -> {v2 - v1:+.3f} W-F1")

    # Frozen baseline anchoring (audio)
    if not np.isnan(audio_wf1.loc["Frozen (IEMOCAP)", "CREMA-D test"]):
        v_frozen = audio_wf1.loc["Frozen (IEMOCAP)", "CREMA-D test"]
        v_ws     = audio_wf1.loc["MELD + CREMA-D + Synth", "CREMA-D test"]
        if not np.isnan(v_ws):
            print(f"  Audio CREMA-D: frozen {v_frozen:.3f} -> "
                  f"with-synth {v_ws:.3f} (delta {v_ws - v_frozen:+.3f})")


if __name__ == "__main__":
    main()
