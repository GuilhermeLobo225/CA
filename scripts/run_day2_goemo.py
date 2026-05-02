#!/usr/bin/env python3
"""
SmartHandover — Day 2: GoEmotions Zero-Shot Evaluation on MELD

Runs GoEmotionsClassifier over all MELD splits using batch inference,
saves predictions to data/processed/goemo_predictions.csv, prints
classification metrics, and compares with the VADER baseline from Day 1.

Usage:
    python run_day2_goemo.py              # Real MELD dataset
    python run_day2_goemo.py --mock       # Dummy data for quick testing
    python run_day2_goemo.py --device 0   # Use GPU 0
"""

import argparse
import os
import sys
import time

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classifiers.goemo_classifier import GoEmotionsClassifier, GOEMO_LABELS

TARGET_LABELS = ["anger", "frustration", "sadness", "neutral", "satisfaction"]
TARGET_LABEL2ID = {label: i for i, label in enumerate(TARGET_LABELS)}

BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Mock data (same as Day 1 for consistency)
# ---------------------------------------------------------------------------

def create_mock_data() -> pd.DataFrame:
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


def load_meld_as_dataframe() -> pd.DataFrame:
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
    parser = argparse.ArgumentParser(description="Day 2 — GoEmotions Zero-Shot on MELD")
    parser.add_argument("--mock", action="store_true", help="Use mock data.")
    parser.add_argument("--device", type=int, default=-1, help="Device: -1=CPU, 0+=GPU")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    print("=" * 60)
    print("  SmartHandover — Day 2: GoEmotions Zero-Shot Baseline")
    print("=" * 60)

    # ----- Load data -----
    if args.mock:
        print("\n[INFO] Using MOCK data.\n")
        df = create_mock_data()
    else:
        print("\n[INFO] Loading MELD dataset from HuggingFace...\n")
        try:
            df = load_meld_as_dataframe()
        except Exception as e:
            print(f"\n[WARN] Failed to load MELD: {e}")
            print("[WARN] Falling back to mock data.\n")
            df = create_mock_data()

    print(f"Total samples: {len(df)}")
    print(f"Class distribution:\n{df['target_emotion'].value_counts().to_string()}\n")

    # ----- Load model -----
    print(f"Loading GoEmotions model (device={args.device})...")
    t0 = time.time()
    classifier = GoEmotionsClassifier(device=args.device, batch_size=args.batch_size)
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    # ----- Batch inference -----
    texts = df["text"].tolist()
    true_labels = df["target_emotion"].tolist()

    print(f"Running batch inference (batch_size={args.batch_size})...")
    all_probs = []
    for i in tqdm(range(0, len(texts), args.batch_size), desc="GoEmotions"):
        batch = texts[i : i + args.batch_size]
        batch_probs = classifier.predict_batch(batch)
        all_probs.extend(batch_probs)

    # ----- Build results DataFrame -----
    rows = []
    for idx, probs in enumerate(all_probs):
        predicted_class = classifier.map_to_target_class(probs)
        row = {
            "text": texts[idx],
            "true_label": true_labels[idx],
            "predicted_class": predicted_class,
        }
        for label in GOEMO_LABELS:
            row[f"goemo_{label}"] = probs.get(label, 0.0)
        rows.append(row)

    results_df = pd.DataFrame(rows)

    # ----- Save CSV -----
    output_path = os.path.join("data", "processed", "goemo_predictions.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")

    # ----- Metrics -----
    y_true = [TARGET_LABEL2ID[l] for l in results_df["true_label"]]
    y_pred = [TARGET_LABEL2ID[l] for l in results_df["predicted_class"]]

    from src.evaluation.metrics import compute_metrics, print_metrics
    metrics = compute_metrics(y_true, y_pred, target_names=TARGET_LABELS)

    print("\n" + "=" * 60)
    print("  GoEmotions Zero-Shot — Results")
    print("=" * 60)
    print_metrics(metrics)

    # ----- Comparison with VADER (Day 1) -----
    vader_path = os.path.join("data", "processed", "vader_predictions.csv")
    if os.path.exists(vader_path):
        print("\n" + "=" * 60)
        print("  Comparison: VADER (Day 1) vs GoEmotions (Day 2)")
        print("=" * 60)

        vader_df = pd.read_csv(vader_path)
        v_true = [TARGET_LABEL2ID[l] for l in vader_df["true_label"]]
        v_pred = [TARGET_LABEL2ID[l] for l in vader_df["predicted_class"]]
        vader_metrics = compute_metrics(v_true, v_pred, target_names=TARGET_LABELS)

        print(f"\n  {'Metric':<20s} {'VADER':>10s} {'GoEmotions':>12s} {'Delta':>10s}")
        print(f"  {'-'*52}")
        for key in ["accuracy", "weighted_f1", "macro_f1", "frustration_recall"]:
            v = vader_metrics[key]
            g = metrics[key]
            delta = g - v
            sign = "+" if delta >= 0 else ""
            print(f"  {key:<20s} {v:>10.4f} {g:>12.4f} {sign}{delta:>9.4f}")
        print()
    else:
        print(f"\n[INFO] VADER predictions not found at {vader_path} — skipping comparison.")

    print(f"Done. Results saved to {output_path}")


if __name__ == "__main__":
    main()
