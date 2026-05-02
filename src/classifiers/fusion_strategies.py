"""
SmartHandover - Multimodal Fusion Strategies (comparative study)

The deliverable rubric explicitly asks (when going multimodal) to compare
at least one fusion strategy with alternatives. The Day 6 meta-classifier
is one strategy ("score fusion"); this module adds two others and
benchmarks all three on the MELD test set:

  Strategy 1 - SCORE FUSION (current baseline)
      Concatenate the per-model probability vectors into a 19-dim feature
      and let a learned meta-classifier (Day 6 MLP) assign the final class.

  Strategy 2 - LATE / WEIGHTED-AVERAGE FUSION
      Re-project every per-model output onto the 5 target classes (with the
      same mapping rules already used during data prep) and take a weighted
      average of the four probability vectors. Weights are grid-searched on
      the validation set.

  Strategy 3 - DECISION / MAJORITY-VOTE FUSION
      Each base model picks an argmax target class; the final prediction is
      the class with most votes. Ties broken by the model with the highest
      individual validation weighted-F1 (=> RoBERTa).

All three strategies share exactly the same per-model probability inputs
that the Day 6 ensemble uses, so the comparison is apples-to-apples and
isolates the effect of the fusion mechanism alone.

Outputs
-------
* Comparison table printed to stdout (markdown + plain).
* ``data/processed/fusion_comparison.csv`` with full metrics per strategy.
* ``data/processed/fusion_comparison.png`` bar chart (W-F1 + frust recall).

Run:

    python -m src.classifiers.fusion_strategies
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.classifiers.ensemble import (  # noqa: E402
    GOEMO_TO_TARGET,
    goemo_row_to_probs,
    vader_to_probs,
)
from src.classifiers.ensemble_trainer import (  # noqa: E402
    CKPT_DIR,
    DATA_DIR,
    FEATURE_COLUMNS,
    TARGET_LABELS,
)
from src.evaluation.metrics import compute_metrics  # noqa: E402

ROBERTA_PROB_COLS = ["prob_anger", "prob_frust", "prob_sadne",
                     "prob_neutr", "prob_satis"]

# IEMOCAP -> 5 target classes (no 'frustration' equivalent in IEMOCAP-ER)
SB_TO_TARGET = {
    "anger":        ["sb_ang"],
    "frustration":  [],            # no audio proxy here on purpose
    "sadness":      ["sb_sad"],
    "neutral":      ["sb_neu"],
    "satisfaction": ["sb_hap"],
}


# ----------------------------------------------------------------------
# Per-model 5-class probability extraction
# ----------------------------------------------------------------------


def _roberta_probs(df: pd.DataFrame) -> np.ndarray:
    return df[ROBERTA_PROB_COLS].to_numpy(dtype=np.float32)


def _goemo_probs(df: pd.DataFrame) -> np.ndarray:
    """Aggregate the 6 GoEmo features into the 5 target classes."""
    out = np.zeros((len(df), 5), dtype=np.float32)
    for i, target in enumerate(TARGET_LABELS):
        for src_label in GOEMO_TO_TARGET[target]:
            col = f"goemo_{src_label}"
            if col in df.columns:
                out[:, i] += df[col].to_numpy(dtype=np.float32)
    # Normalise so each row sums to 1 (some surprise/disgust columns may be
    # missing in older feature CSVs - guard against zero rows).
    sums = out.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return out / sums


def _vader_probs(df: pd.DataFrame) -> np.ndarray:
    return np.vstack([vader_to_probs(c) for c in df["vader_compound"].values])


def _sb_probs(df: pd.DataFrame) -> np.ndarray:
    """Map IEMOCAP audio probs (4 classes) onto the 5 target classes.

    IEMOCAP has no frustration, so the audio model gives 0 mass to that
    class - this is by design. The remaining four target classes get
    their corresponding audio probability directly.
    """
    out = np.zeros((len(df), 5), dtype=np.float32)
    for i, target in enumerate(TARGET_LABELS):
        for col in SB_TO_TARGET[target]:
            if col in df.columns:
                out[:, i] += df[col].to_numpy(dtype=np.float32)
    return out


def _all_probs(split_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {
        "roberta":     _roberta_probs(split_df),
        "goemo":       _goemo_probs(split_df),
        "vader":       _vader_probs(split_df),
        "speechbrain": _sb_probs(split_df),
    }


# ----------------------------------------------------------------------
# Strategy 1 - score fusion (existing meta-classifier)
# ----------------------------------------------------------------------


def score_fusion_predict(test_df: pd.DataFrame) -> np.ndarray:
    """Use the Day-6 meta-classifier (concat features -> MLP)."""
    bundle = joblib.load(os.path.join(CKPT_DIR, "meta_classifier.pkl"))
    model = bundle["model"]
    feat_cols = bundle.get("feature_columns", FEATURE_COLUMNS)
    X = test_df[feat_cols].to_numpy(dtype=np.float32)
    return model.predict(X)


# ----------------------------------------------------------------------
# Strategy 2 - late / weighted-average fusion
# ----------------------------------------------------------------------


def _weighted_average(probs: Dict[str, np.ndarray],
                      weights: Dict[str, float]) -> np.ndarray:
    total_w = sum(weights.values())
    if total_w <= 0:
        raise ValueError("Sum of fusion weights must be positive")
    fused = np.zeros_like(probs["roberta"])
    for k, w in weights.items():
        fused += (w / total_w) * probs[k]
    return fused


def _grid_search_weights(val_probs: Dict[str, np.ndarray],
                         y_val: np.ndarray) -> Tuple[Dict[str, float], float]:
    """Coarse grid search on validation weighted F1.

    The combinatorics are kept small (4 weights x 4 levels each) to stay
    well under a second.
    """
    levels = [0.1, 0.3, 0.5, 0.7]
    best_w, best_f1 = None, -1.0
    for wr in levels:
        for wg in levels:
            for wv in levels:
                for ws in levels:
                    weights = {
                        "roberta": wr, "goemo": wg,
                        "vader": wv, "speechbrain": ws,
                    }
                    fused = _weighted_average(val_probs, weights)
                    preds = fused.argmax(axis=1)
                    m = compute_metrics(y_val, preds, target_names=TARGET_LABELS)
                    if m["weighted_f1"] > best_f1:
                        best_f1 = m["weighted_f1"]
                        best_w = weights
    return best_w, best_f1


def late_fusion_predict(test_probs: Dict[str, np.ndarray],
                        val_probs: Dict[str, np.ndarray],
                        y_val: np.ndarray
                        ) -> Tuple[np.ndarray, Dict[str, float]]:
    weights, val_f1 = _grid_search_weights(val_probs, y_val)
    fused = _weighted_average(test_probs, weights)
    return fused.argmax(axis=1), {"weights": weights, "val_w_f1": val_f1}


# ----------------------------------------------------------------------
# Strategy 3 - decision / majority-vote fusion
# ----------------------------------------------------------------------


def majority_vote_predict(test_probs: Dict[str, np.ndarray],
                          tie_break_order: List[str]) -> np.ndarray:
    """Each model picks its argmax; majority class wins, ties broken by the
    model that ranks first in ``tie_break_order``.
    """
    n = test_probs["roberta"].shape[0]
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        votes = np.zeros(len(TARGET_LABELS), dtype=np.int32)
        first_voter: Dict[int, str] = {}
        for model_name in tie_break_order:
            cls = int(np.argmax(test_probs[model_name][i]))
            votes[cls] += 1
            first_voter.setdefault(cls, model_name)

        max_votes = votes.max()
        winners = np.flatnonzero(votes == max_votes)
        if len(winners) == 1:
            out[i] = winners[0]
        else:
            # Tie - pick the class voted for by the highest-ranked model
            for model_name in tie_break_order:
                cls = int(np.argmax(test_probs[model_name][i]))
                if cls in winners:
                    out[i] = cls
                    break
    return out


# ----------------------------------------------------------------------
# Plotting / reporting
# ----------------------------------------------------------------------


def _bar_chart(df: pd.DataFrame, out_path: str) -> None:
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    palette = sns.color_palette("crest", n_colors=len(df))

    sns.barplot(data=df, x="strategy", y="weighted_f1",
                ax=axes[0], palette=palette)
    axes[0].set_title("Weighted F1 (MELD test)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Weighted F1")
    for p in axes[0].patches:
        axes[0].text(p.get_x() + p.get_width() / 2, p.get_height() + 0.005,
                     f"{p.get_height():.3f}", ha="center", va="bottom",
                     fontsize=10)

    sns.barplot(data=df, x="strategy", y="frust_recall",
                ax=axes[1], palette=palette)
    axes[1].set_title("Frustration Recall (MELD test)")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Frustration Recall")
    for p in axes[1].patches:
        axes[1].text(p.get_x() + p.get_width() / 2, p.get_height() + 0.005,
                     f"{p.get_height():.3f}", ha="center", va="bottom",
                     fontsize=10)

    fig.suptitle("SmartHandover - Fusion-Strategy Comparison",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> None:
    print("=" * 72)
    print("  SmartHandover - Fusion-Strategy Comparison")
    print("=" * 72)

    val_df  = pd.read_csv(os.path.join(DATA_DIR, "ensemble_features_val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "ensemble_features_test.csv"))
    y_val  = val_df["true_label_id"].to_numpy()
    y_test = test_df["true_label_id"].to_numpy()

    val_probs  = _all_probs(val_df)
    test_probs = _all_probs(test_df)

    # ----- Per-model individual baseline (sanity, for tie-break ordering)
    print("\n[1/4] Per-model validation weighted F1 (used for tie-break order):")
    rankings: List[Tuple[str, float]] = []
    for name, P in val_probs.items():
        preds = P.argmax(axis=1)
        wf1 = compute_metrics(y_val, preds, target_names=TARGET_LABELS)["weighted_f1"]
        rankings.append((name, wf1))
        print(f"    {name:<13s}  W-F1={wf1:.4f}")
    rankings.sort(key=lambda kv: -kv[1])
    tie_order = [name for name, _ in rankings]
    print(f"  Tie-break order: {tie_order}")

    # ----- Strategy 1: Score fusion (existing MLP)
    print("\n[2/4] Strategy 1 - Score fusion (Day-6 MLP)")
    s1_preds = score_fusion_predict(test_df)
    s1_m = compute_metrics(y_test, s1_preds, target_names=TARGET_LABELS)

    # ----- Strategy 2: Late/weighted-average fusion
    print("\n[3/4] Strategy 2 - Late fusion (weighted-average, grid-searched)")
    s2_preds, s2_meta = late_fusion_predict(test_probs, val_probs, y_val)
    s2_m = compute_metrics(y_test, s2_preds, target_names=TARGET_LABELS)
    print(f"  best weights -> {s2_meta['weights']}")
    print(f"  val W-F1     -> {s2_meta['val_w_f1']:.4f}")

    # ----- Strategy 3: Decision/majority-vote fusion
    print("\n[4/4] Strategy 3 - Majority-vote fusion")
    s3_preds = majority_vote_predict(test_probs, tie_order)
    s3_m = compute_metrics(y_test, s3_preds, target_names=TARGET_LABELS)

    rows = []
    for name, m in [("score_fusion (MLP)", s1_m),
                    ("late_fusion (avg)",  s2_m),
                    ("decision_fusion (vote)", s3_m)]:
        rows.append({
            "strategy":          name,
            "accuracy":          m["accuracy"],
            "weighted_f1":       m["weighted_f1"],
            "macro_f1":          m["macro_f1"],
            "frust_recall":      m["frustration_recall"],
            "frust_f1":          m["per_class"].get("frustration", {}).get("f1", 0.0),
        })
    df = pd.DataFrame(rows)

    print("\n" + "=" * 72)
    print("  Fusion Comparison (MELD test)")
    print("=" * 72)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # CSV + chart
    out_csv = os.path.join(DATA_DIR, "fusion_comparison.csv")
    out_png = os.path.join(DATA_DIR, "fusion_comparison.png")
    df.to_csv(out_csv, index=False)
    _bar_chart(df, out_png)
    print(f"\n  CSV   -> {out_csv}")
    print(f"  Chart -> {out_png}")

    # Save the late-fusion weights for reproducibility
    import json
    meta_path = os.path.join(DATA_DIR, "fusion_late_weights.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"weights": s2_meta["weights"],
                   "tie_break_order": tie_order}, f, indent=2)
    print(f"  Late-fusion weights saved -> {meta_path}")


if __name__ == "__main__":
    main()
