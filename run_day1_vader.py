#!/usr/bin/env python3
"""
SmartHandover — Day 1: VADER Baseline Evaluation on MELD

Runs the VaderClassifier over all MELD splits (train, validation, test),
saves predictions to data/processed/vader_predictions.csv, and prints
classification metrics (accuracy, weighted F1, macro F1, confusion matrix).

Usage:
    python run_day1_vader.py            # Uses real MELD dataset (requires download)
    python run_day1_vader.py --mock     # Uses mock/dummy data for quick testing
"""

import argparse
import os
import sys

import pandas as pd
from tqdm import tqdm

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.classifiers.vader_classifier import VaderClassifier

# Target classes (defined here to avoid importing load_meld at module level,
# which requires the `datasets` library even when using --mock mode)
TARGET_LABELS = ["anger", "frustration", "sadness", "neutral", "satisfaction"]
TARGET_LABEL2ID = {label: i for i, label in enumerate(TARGET_LABELS)}


# ---------------------------------------------------------------------------
# Mock data generator (fallback when MELD is unavailable)
# ---------------------------------------------------------------------------

def create_mock_data() -> pd.DataFrame:
    """Generate a small dummy DataFrame mimicking MELD structure."""
    samples = [
        ("I am so angry at you right now!", "anger"),
        ("This is absolutely unacceptable, I'm furious!", "anger"),
        ("You ruined everything, I hate this!", "anger"),
        ("Stop doing that, it's driving me crazy!", "anger"),
        ("I can't believe you did that, I'm livid!", "anger"),
        ("This is so frustrating, nothing works.", "frustration"),
        ("I've been waiting for an hour, this is ridiculous.", "frustration"),
        ("Why does this keep happening to me?", "frustration"),
        ("I'm so tired of dealing with this problem.", "frustration"),
        ("Every time I try, something goes wrong.", "frustration"),
        ("I feel so sad and lonely today.", "sadness"),
        ("It breaks my heart to see this.", "sadness"),
        ("I miss the way things used to be.", "sadness"),
        ("Nothing seems to matter anymore.", "sadness"),
        ("I just want to cry.", "sadness"),
        ("The meeting is at 3pm.", "neutral"),
        ("I'll check and get back to you.", "neutral"),
        ("Sure, that works for me.", "neutral"),
        ("Let me know when you're ready.", "neutral"),
        ("The report is on the shared drive.", "neutral"),
        ("Okay.", "neutral"),
        ("I see.", "neutral"),
        ("That's fine.", "neutral"),
        ("This is wonderful, I'm so happy!", "satisfaction"),
        ("Great job, I really appreciate your help!", "satisfaction"),
        ("Everything worked out perfectly!", "satisfaction"),
        ("I'm really pleased with the results.", "satisfaction"),
        ("Thank you so much, this is amazing!", "satisfaction"),
    ]
    return pd.DataFrame(samples, columns=["text", "target_emotion"])


# ---------------------------------------------------------------------------
# Load real MELD data into a pandas DataFrame
# ---------------------------------------------------------------------------

def load_meld_as_dataframe() -> pd.DataFrame:
    """Load all MELD splits and return a single DataFrame."""
    from src.data.load_meld import load_meld

    rows = []
    for split in ["train", "validation", "test"]:
        print(f"  Loading MELD split: {split} ...")
        ds = load_meld(split=split, streaming=False)
        for example in ds:
            rows.append({
                "text": example["text"],
                "target_emotion": example["target_emotion"],
                "split": split,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Day 1 — VADER Baseline on MELD")
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock/dummy data instead of the real MELD dataset."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  SmartHandover — Day 1: VADER Baseline")
    print("=" * 60)

    # ----- Load data -----
    if args.mock:
        print("\n[INFO] Using MOCK data (--mock flag).\n")
        df = create_mock_data()
    else:
        print("\n[INFO] Loading real MELD dataset from HuggingFace...\n")
        try:
            df = load_meld_as_dataframe()
        except Exception as e:
            print(f"\n[WARN] Failed to load MELD dataset: {e}")
            print("[WARN] Falling back to mock data.\n")
            df = create_mock_data()

    print(f"\nTotal samples: {len(df)}")
    print(f"Class distribution:\n{df['target_emotion'].value_counts().to_string()}\n")

    # ----- Run VADER -----
    print("Running VADER classifier on all samples...")
    classifier = VaderClassifier()

    predictions = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="VADER"):
        result = classifier.predict_and_classify(row["text"])
        predictions.append({
            "text": row["text"],
            "true_label": row["target_emotion"],
            "predicted_class": result["predicted_class"],
            "vader_pos": result["pos"],
            "vader_neg": result["neg"],
            "vader_neu": result["neu"],
            "vader_compound": result["compound"],
        })

    results_df = pd.DataFrame(predictions)

    # ----- Save CSV -----
    output_path = os.path.join("data", "processed", "vader_predictions.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")

    # ----- Compute metrics -----
    # Convert string labels to integer indices for metrics.py
    y_true = [TARGET_LABEL2ID[label] for label in results_df["true_label"]]
    y_pred = [TARGET_LABEL2ID[label] for label in results_df["predicted_class"]]

    # Import metrics inline to avoid pulling datasets at module level
    from src.evaluation.metrics import compute_metrics, print_metrics
    metrics = compute_metrics(y_true, y_pred, target_names=TARGET_LABELS)

    print("\n" + "=" * 60)
    print("  VADER Baseline — Results")
    print("=" * 60)
    print_metrics(metrics)

    print(f"\nDone. Results saved to {output_path}")


if __name__ == "__main__":
    main()
