"""
SmartHandover — Ablation Study (Day 7)

Measures the individual contribution of each of the four component models
(RoBERTa, GoEmotions, VADER, SpeechBrain) to the final ensemble.

For each feature subset we:
  1. Fit the same classifier architecture used by the full meta-classifier
     (LogisticRegression with class_weight='balanced' — fast, deterministic,
     a faithful proxy for per-component contribution).
  2. Score weighted F1, macro F1 and frustration recall on the MELD test set.

Outputs
-------
* Markdown-formatted comparison table printed to stdout.
* Bar chart (W-F1 and Frustration Recall) saved to
  ``data/processed/ablation_results.png``.

Run:

    python -m src.evaluation.ablation
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # Headless-safe backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

# Project imports -----------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.classifiers.ensemble_trainer import (  # noqa: E402
    DATA_DIR,
    FEATURE_COLUMNS,
    GOEMO_COLS,
    ROBERTA_COLS,
    SB_COLS,
    TARGET_LABELS,
    VADER_COLS,
)
from src.evaluation.metrics import compute_metrics  # noqa: E402

# ---------------------------------------------------------------------------
# Feature-subset configurations
# ---------------------------------------------------------------------------

COMPONENT_COLS: Dict[str, List[str]] = {
    "RoBERTa": ROBERTA_COLS,
    "GoEmotions": GOEMO_COLS,
    "VADER": VADER_COLS,
    "SpeechBrain": SB_COLS,
}

# (config_name, list_of_component_keys_to_keep)
ABLATION_CONFIGS: List[Tuple[str, List[str]]] = [
    ("Full ensemble",    ["RoBERTa", "GoEmotions", "VADER", "SpeechBrain"]),
    ("No VADER",         ["RoBERTa", "GoEmotions", "SpeechBrain"]),
    ("No GoEmotions",    ["RoBERTa", "VADER", "SpeechBrain"]),
    ("No SpeechBrain",   ["RoBERTa", "GoEmotions", "VADER"]),
    ("No RoBERTa",       ["GoEmotions", "VADER", "SpeechBrain"]),
    ("Only RoBERTa",     ["RoBERTa"]),
    ("Only SpeechBrain", ["SpeechBrain"]),
    ("Only GoEmotions",  ["GoEmotions"]),
    ("Only VADER",       ["VADER"]),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _subset_columns(components: List[str]) -> List[str]:
    """Return the feature columns that belong to the requested components."""
    cols: List[str] = []
    for comp in components:
        if comp not in COMPONENT_COLS:
            raise KeyError(f"Unknown component: {comp}")
        cols.extend(COMPONENT_COLS[comp])
    # Keep canonical order (as defined in FEATURE_COLUMNS)
    return [c for c in FEATURE_COLUMNS if c in set(cols)]


def _load_features() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(os.path.join(DATA_DIR, "ensemble_features_train.csv"))
    val = pd.read_csv(os.path.join(DATA_DIR, "ensemble_features_val.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "ensemble_features_test.csv"))
    return train, val, test


def _make_classifier() -> LogisticRegression:
    """Factory for the ablation meta-classifier.

    LogisticRegression is used here (rather than the MLP chosen in Day 6)
    for speed & determinism — the goal is to quantify feature contribution,
    not to hit peak absolute performance.
    """
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------


def run_ablation() -> pd.DataFrame:
    """Run all configurations and return a results DataFrame (sorted)."""
    train, val, test = _load_features()

    y_train = train["true_label_id"].to_numpy()
    y_val = val["true_label_id"].to_numpy()
    y_test = test["true_label_id"].to_numpy()

    rows = []
    for name, components in ABLATION_CONFIGS:
        cols = _subset_columns(components)
        X_train = train[cols].to_numpy(dtype=np.float32)
        X_val = val[cols].to_numpy(dtype=np.float32)
        X_test = test[cols].to_numpy(dtype=np.float32)

        clf = _make_classifier()
        clf.fit(X_train, y_train)

        val_preds = clf.predict(X_val)
        test_preds = clf.predict(X_test)

        val_metrics = compute_metrics(y_val, val_preds, target_names=TARGET_LABELS)
        test_metrics = compute_metrics(y_test, test_preds, target_names=TARGET_LABELS)

        rows.append({
            "Configuration":     name,
            "N_features":        len(cols),
            "Val_W-F1":          val_metrics["weighted_f1"],
            "Test_W-F1":         test_metrics["weighted_f1"],
            "Test_M-F1":         test_metrics["macro_f1"],
            "Test_Frust_Recall": test_metrics["frustration_recall"],
        })

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Reporting: markdown table + bar chart
# ---------------------------------------------------------------------------


def to_markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-style markdown table."""
    cols = df.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def plot_ablation(df: pd.DataFrame, out_path: str) -> None:
    """Save a side-by-side bar chart of test W-F1 and frustration recall."""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    order = df.sort_values("Test_W-F1", ascending=False)["Configuration"].tolist()
    palette = sns.color_palette("viridis", n_colors=len(order))

    # ---- Panel 1: Weighted F1 ----
    sns.barplot(
        data=df, x="Configuration", y="Test_W-F1",
        order=order, palette=palette, ax=axes[0],
    )
    axes[0].set_title("Ablation — Weighted F1 (MELD test)")
    axes[0].set_ylabel("Weighted F1")
    axes[0].set_xlabel("")
    axes[0].set_ylim(0, max(df["Test_W-F1"].max() * 1.15, 0.1))
    axes[0].tick_params(axis="x", rotation=35)
    for lbl in axes[0].get_xticklabels():
        lbl.set_ha("right")
    for p in axes[0].patches:
        h = p.get_height()
        axes[0].text(p.get_x() + p.get_width() / 2, h + 0.005,
                     f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    # ---- Panel 2: Frustration recall ----
    sns.barplot(
        data=df, x="Configuration", y="Test_Frust_Recall",
        order=order, palette=palette, ax=axes[1],
    )
    axes[1].set_title("Ablation — Frustration Recall (MELD test)")
    axes[1].set_ylabel("Frustration Recall")
    axes[1].set_xlabel("")
    axes[1].set_ylim(0, max(df["Test_Frust_Recall"].max() * 1.25, 0.1))
    axes[1].tick_params(axis="x", rotation=35)
    for lbl in axes[1].get_xticklabels():
        lbl.set_ha("right")
    for p in axes[1].patches:
        h = p.get_height()
        axes[1].text(p.get_x() + p.get_width() / 2, h + 0.005,
                     f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("SmartHandover — Ablation Study", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("  SmartHandover — Day 7: Ablation Study")
    print("=" * 60)

    df = run_ablation()
    df_sorted = df.sort_values("Test_W-F1", ascending=False).reset_index(drop=True)

    # Markdown table to stdout
    print("\n### Ablation Results (sorted by Test Weighted F1)\n")
    print(to_markdown_table(df_sorted))

    # Derived insights
    full = df.set_index("Configuration").loc["Full ensemble"]
    print("\n### Component Contribution (vs Full ensemble)\n")
    print(f"  Full ensemble test W-F1: {full['Test_W-F1']:.4f} "
          f"(Frust-R={full['Test_Frust_Recall']:.4f})\n")

    for leave_out in ["RoBERTa", "GoEmotions", "VADER", "SpeechBrain"]:
        name = f"No {leave_out}"
        row = df.set_index("Configuration").loc[name]
        delta_wf1 = row["Test_W-F1"] - full["Test_W-F1"]
        delta_fr = row["Test_Frust_Recall"] - full["Test_Frust_Recall"]
        print(f"  Remove {leave_out:<12s} -> "
              f"delta W-F1 = {delta_wf1:+.4f}, "
              f"delta Frust-R = {delta_fr:+.4f}")

    # Persist CSV + chart
    out_csv = os.path.join(DATA_DIR, "ablation_results.csv")
    df_sorted.to_csv(out_csv, index=False)
    print(f"\n  CSV saved   -> {out_csv}")

    out_png = os.path.join(DATA_DIR, "ablation_results.png")
    plot_ablation(df, out_png)
    print(f"  Chart saved -> {out_png}")

    print("\nDone.")


if __name__ == "__main__":
    main()
