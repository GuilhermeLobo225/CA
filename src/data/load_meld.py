"""
Smart Handover - MELD Dataset Ingestion Script
Phase 1: English Language

Uses the ajyy/MELD_audio dataset on HuggingFace, which provides pre-extracted
16 kHz mono FLAC audio + text from the original MELD dataset.

Original MELD classes: anger, disgust, fear, joy, neutral, sadness, surprise
Target classes:        anger, frustration, sadness, neutral, satisfaction

Mapping:
  anger    -> anger
  disgust  -> anger        (merged: disgust is anger-adjacent)
  fear     -> frustration  (proxy: fear maps to frustration in service contexts)
  joy      -> satisfaction
  neutral  -> neutral
  sadness  -> sadness
  surprise -> DROPPED      (ambiguous valence, not useful for frustration detection)

Dataset source: https://huggingface.co/datasets/ajyy/MELD_audio
Features:  text (str), audio (Audio@16kHz), emotion (str), sentiment (str)
"""

from datasets import load_dataset

MELD_LABEL_MAP = {
    "anger": "anger",
    "disgust": "anger",
    "fear": "frustration",
    "joy": "satisfaction",
    "neutral": "neutral",
    "sadness": "sadness",
    "surprise": None,  # drop
}

TARGET_LABELS = ["anger", "frustration", "sadness", "neutral", "satisfaction"]
TARGET_LABEL2ID = {label: i for i, label in enumerate(TARGET_LABELS)}
TARGET_ID2LABEL = {i: label for i, label in enumerate(TARGET_LABELS)}

# HuggingFace dataset identifier
_HF_DATASET = "ajyy/MELD_audio"


def load_meld(split: str = "train", streaming: bool = False):
    """Load MELD audio dataset from HuggingFace and remap emotions.

    Args:
        split: One of 'train', 'validation', 'test'.
        streaming: If True, returns an iterable dataset (lower memory).

    Returns:
        A HuggingFace Dataset with remapped 'target_emotion' and 'target_label'
        columns.  Rows originally labelled 'surprise' are removed.
        Audio is available at 16 kHz via the 'audio' column.
    """
    ds = load_dataset(_HF_DATASET, split=split, streaming=streaming, trust_remote_code=True)

    def remap(example):
        # emotion is a plain string in ajyy/MELD_audio
        original_str = example["emotion"].lower().strip()
        mapped = MELD_LABEL_MAP.get(original_str)
        example["original_emotion"] = original_str
        example["target_emotion"] = mapped if mapped else "DROP"
        example["target_label"] = TARGET_LABEL2ID[mapped] if mapped else -1
        return example

    ds = ds.map(remap)

    # Drop surprise (target_label == -1)
    ds = ds.filter(lambda x: x["target_label"] != -1)

    return ds


def load_all_splits(streaming: bool = False):
    """Load train, validation, and test splits with remapped labels."""
    return {
        split: load_meld(split, streaming=streaming)
        for split in ["train", "validation", "test"]
    }


def print_split_stats(ds, split_name: str):
    """Print class distribution for a loaded split."""
    from collections import Counter

    counts = Counter(ds["target_emotion"])
    total = sum(counts.values())
    print(f"\n{'='*50}")
    print(f"  Split: {split_name} | Total samples: {total}")
    print(f"{'='*50}")
    for label in TARGET_LABELS:
        c = counts.get(label, 0)
        pct = 100 * c / total if total else 0
        bar = "█" * int(pct / 2)
        print(f"  {label:<15s} {c:>5d}  ({pct:5.1f}%)  {bar}")


if __name__ == "__main__":
    print("=" * 50)
    print("  Smart Handover — MELD Dataset Loader")
    print(f"  Source: {_HF_DATASET}")
    print("=" * 50)
    print("\nDownloading and loading MELD audio dataset...")
    splits = load_all_splits(streaming=False)

    for name, ds in splits.items():
        print_split_stats(ds, name)

    # Quick sanity check on first sample
    sample = splits["train"][0]
    print(f"\n--- Sample (train[0]) ---")
    print(f"  Text           : {sample['text'][:80]}...")
    print(f"  Original emotion: {sample['original_emotion']}")
    print(f"  Target emotion  : {sample['target_emotion']}")
    print(f"  Target label    : {sample['target_label']}")
    print(f"  Audio sample rate: {sample['audio']['sampling_rate']} Hz")
    print(f"  Audio length     : {len(sample['audio']['array']) / sample['audio']['sampling_rate']:.2f}s")

    print("\nDone. Dataset loaded and remapped successfully.")
