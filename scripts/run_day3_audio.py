#!/usr/bin/env python3
"""
SmartHandover — Day 3: SpeechBrain Audio Emotion + Whisper ASR on MELD

Part A: Runs SpeechBrainClassifier on MELD audio, saves predictions, prints metrics.
Part B: Runs WhisperASR on a 50-sample subset, computes WER vs original text.

Manages VRAM by unloading SpeechBrain before loading Whisper.

Usage:
    python run_day3_audio.py                # Real MELD + GPU
    python run_day3_audio.py --mock         # Dummy data (no download)
    python run_day3_audio.py --device cpu   # Force CPU
    python run_day3_audio.py --skip-whisper # Only run Part A
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET_LABELS = ["anger", "frustration", "sadness", "neutral", "satisfaction"]
TARGET_LABEL2ID = {label: i for i, label in enumerate(TARGET_LABELS)}

# SpeechBrain maps ang->anger, but frustration has no direct SpeechBrain class.
# For metric computation we map SpeechBrain's "anger" prediction to label index 0.
SB_TARGET_MAP = {
    "anger": 0,
    "satisfaction": 4,
    "sadness": 2,
    "neutral": 3,
}


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

def create_mock_data() -> pd.DataFrame:
    """Generate dummy audio + text data for testing."""
    rng = np.random.default_rng(42)
    samples = []
    labels = ["anger", "anger", "frustration", "frustration",
              "sadness", "sadness", "neutral", "neutral", "neutral",
              "satisfaction", "satisfaction"]
    texts = [
        "I am so angry at you right now!",
        "This is absolutely unacceptable!",
        "This is so frustrating, nothing works.",
        "I've been waiting for an hour.",
        "I feel so sad and lonely today.",
        "It breaks my heart to see this.",
        "The meeting is at 3pm.",
        "Sure, that works for me.",
        "Okay.",
        "This is wonderful, I'm so happy!",
        "Great job, I really appreciate your help!",
    ]
    for i, (text, label) in enumerate(zip(texts, labels)):
        # Fake audio: 2 seconds of random noise at 16kHz
        audio_array = rng.standard_normal(16000 * 2).astype(np.float32) * 0.01
        samples.append({
            "audio_id": f"mock_{i:04d}",
            "text": text,
            "target_emotion": label,
            "audio_array": audio_array,
            "sr": 16000,
        })
    return pd.DataFrame(samples)


def load_meld_audio_dataframe(max_samples: int = None) -> pd.DataFrame:
    """Load MELD with audio arrays into a DataFrame."""
    from src.data.load_meld import load_meld

    rows = []
    for split in ["train", "validation", "test"]:
        print(f"  Loading MELD split: {split} ...")
        ds = load_meld(split=split, streaming=False)
        for idx, example in enumerate(ds):
            rows.append({
                "audio_id": f"{split}_{idx:05d}",
                "text": example["text"],
                "target_emotion": example["target_emotion"],
                "audio_array": np.array(example["audio"]["array"], dtype=np.float32),
                "sr": example["audio"]["sampling_rate"],
            })
            if max_samples and len(rows) >= max_samples:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part A: SpeechBrain Emotion from Audio
# ---------------------------------------------------------------------------

def run_speechbrain(df: pd.DataFrame, device: str) -> pd.DataFrame:
    """Run SpeechBrain on all audio samples."""
    from src.classifiers.speechbrain_classifier import SpeechBrainClassifier, IEMOCAP_LABELS, IEMOCAP_TO_TARGET

    print(f"\n[Part A] Loading SpeechBrain model (device={device})...")
    t0 = time.time()
    sb = SpeechBrainClassifier(device=device)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="SpeechBrain"):
        res = sb.predict_and_classify(row["audio_array"], sr=row["sr"])
        results.append({
            "audio_id": row["audio_id"],
            "true_label": row["target_emotion"],
            "predicted_class": res["predicted_class"],
            "sb_ang": res["ang"],
            "sb_hap": res["hap"],
            "sb_sad": res["sad"],
            "sb_neu": res["neu"],
        })

    # Free VRAM
    del sb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  SpeechBrain unloaded, VRAM freed.")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Part B: Whisper ASR + WER
# ---------------------------------------------------------------------------

def run_whisper_wer(df: pd.DataFrame, device: str, n_samples: int = 50):
    """Transcribe a subset with Whisper and compute WER."""
    from src.classifiers.whisper_asr import WhisperASR
    from jiwer import wer as compute_wer

    subset = df.head(n_samples).copy()

    print(f"\n[Part B] Loading Whisper (device={device})...")
    t0 = time.time()
    asr = WhisperASR(model_size="small", device=device)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    references = []
    hypotheses = []

    for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Whisper ASR"):
        transcription = asr.transcribe(row["audio_array"], sr=row["sr"])
        references.append(row["text"])
        hypotheses.append(transcription)

    # Free VRAM
    del asr
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Compute WER
    overall_wer = compute_wer(references, hypotheses)

    print(f"\n  {'='*50}")
    print(f"  Whisper ASR — Word Error Rate (n={len(subset)})")
    print(f"  {'='*50}")
    print(f"  WER: {overall_wer:.4f} ({overall_wer*100:.1f}%)")
    target = "PASS" if overall_wer < 0.15 else "FAIL"
    print(f"  Target < 15%: {target}")

    # Show a few examples
    print(f"\n  Sample transcriptions:")
    for i in range(min(5, len(references))):
        ref = references[i][:80].encode("ascii", errors="replace").decode()
        hyp = hypotheses[i][:80].encode("ascii", errors="replace").decode()
        print(f"  [{i}] REF: {ref}")
        print(f"       HYP: {hyp}")
        print()

    return overall_wer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Day 3 — Audio: SpeechBrain + Whisper")
    parser.add_argument("--mock", action="store_true", help="Use mock data.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cpu' or 'cuda' (auto-detects if omitted)")
    parser.add_argument("--skip-whisper", action="store_true",
                        help="Skip Whisper ASR (Part B)")
    parser.add_argument("--whisper-samples", type=int, default=50,
                        help="Number of samples for Whisper WER test")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  SmartHandover — Day 3: Audio Pipeline")
    print(f"  Device: {device}")
    print("=" * 60)

    # ----- Load data -----
    if args.mock:
        print("\n[INFO] Using MOCK data.\n")
        df = create_mock_data()
    else:
        print("\n[INFO] Loading MELD with audio...\n")
        try:
            df = load_meld_audio_dataframe()
        except Exception as e:
            print(f"\n[WARN] Failed to load MELD audio: {e}")
            print("[WARN] Falling back to mock data.\n")
            df = create_mock_data()

    print(f"Total samples: {len(df)}")
    print(f"Class distribution:\n{df['target_emotion'].value_counts().to_string()}\n")

    # ===== PART A: SpeechBrain =====
    sb_results = run_speechbrain(df, device)

    # Save CSV
    sb_path = os.path.join("data", "processed", "speechbrain_predictions.csv")
    os.makedirs(os.path.dirname(sb_path), exist_ok=True)
    sb_results.to_csv(sb_path, index=False)
    print(f"\n  SpeechBrain predictions saved to: {sb_path}")

    # Metrics — note: SpeechBrain never predicts "frustration" directly,
    # so we compute metrics only over the 4 classes it can produce.
    from src.evaluation.metrics import compute_metrics, print_metrics

    y_true = [TARGET_LABEL2ID[l] for l in sb_results["true_label"]]
    y_pred = [SB_TARGET_MAP[l] for l in sb_results["predicted_class"]]

    metrics = compute_metrics(y_true, y_pred, target_names=TARGET_LABELS)

    print("\n" + "=" * 60)
    print("  SpeechBrain Audio — Results")
    print("=" * 60)
    print_metrics(metrics)

    # ===== PART B: Whisper WER =====
    if not args.skip_whisper:
        run_whisper_wer(df, device, n_samples=args.whisper_samples)
    else:
        print("\n[INFO] Whisper ASR skipped (--skip-whisper).")

    # ===== Comparison table (all 3 baselines) =====
    print("\n" + "=" * 60)
    print("  Baseline Comparison (Day 1-3)")
    print("=" * 60)

    rows = [("SpeechBrain (audio)", metrics["weighted_f1"], metrics["macro_f1"],
             metrics["frustration_recall"])]

    for name, path in [("VADER", "data/processed/vader_predictions.csv"),
                       ("GoEmotions", "data/processed/goemo_predictions.csv")]:
        if os.path.exists(path):
            prev_df = pd.read_csv(path)
            yt = [TARGET_LABEL2ID[l] for l in prev_df["true_label"]]
            yp = [TARGET_LABEL2ID[l] for l in prev_df["predicted_class"]]
            m = compute_metrics(yt, yp, target_names=TARGET_LABELS)
            rows.append((name, m["weighted_f1"], m["macro_f1"], m["frustration_recall"]))

    print(f"\n  {'Model':<22s} {'W-F1':>8s} {'M-F1':>8s} {'Frust-R':>8s}")
    print(f"  {'-'*48}")
    for name, wf1, mf1, fr in sorted(rows, key=lambda x: -x[1]):
        print(f"  {name:<22s} {wf1:>8.4f} {mf1:>8.4f} {fr:>8.4f}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
