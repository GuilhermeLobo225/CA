"""
SmartHandover - CREMA-D Dataset Loader

CREMA-D (Crowd-Sourced Emotional Multimodal Actors Dataset)
-----------------------------------------------------------
* 7 442 audio clips, 91 actors (48 M + 43 F, ages 20-74)
* 12 lexically-neutral fixed sentences
* 6 emotions: anger, disgust, fear, happy, neutral, sad
* 4 intensity levels: low / medium / high / unspecified
* 16-bit / 16 kHz / mono WAV
* License: Open Database License (ODbL) v1.0 - completely public
* Source: https://github.com/CheyneyComputerScience/CREMA-D

Filename pattern (in AudioWAV/):
    1001_DFA_ANG_XX.wav
    ----  ---  ---  --
    actor sent emo  intensity (LO/MD/HI/XX)

Mapping to SmartHandover 5 target classes (default):

    ANG -> anger
    DIS -> anger          (proxy, consistent with MELD's disgust mapping)
    FEA -> DROPPED        (acted fear is not customer frustration; opt-in
                           via fear_as_frustration=True for consistency
                           with the existing MELD load behaviour)
    HAP -> satisfaction
    NEU -> neutral
    SAD -> sadness

Because every CREMA-D utterance is one of 12 fixed sentences, this dataset
is **purely an acoustic asset**. The text is identical across emotions
within a sentence ID and therefore useless for fine-tuning a text-only
classifier - the loader still returns the canonical sentence string so
that downstream code (e.g. the meta-classifier feature builder) can keep
its existing column layout.

Download
--------
This loader does not pull from the internet. Place the CREMA-D files at:

    data/raw/CREMA-D/AudioWAV/*.wav

Either:

  (1) Clone the upstream repo:
        git clone https://github.com/CheyneyComputerScience/CREMA-D.git data/raw/CREMA-D
  (2) Or download just the AudioWAV folder (~580 MB) from the GitHub
      release and unzip it under data/raw/CREMA-D/.

The optional ``VideoDemographics.csv`` (under the same repo) provides
actor metadata - if present, the loader joins it onto each row.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CREMAD_LABEL_MAP_DEFAULT: Dict[str, Optional[str]] = {
    "ANG": "anger",
    "DIS": "anger",          # proxy
    "FEA": None,             # dropped by default
    "HAP": "satisfaction",
    "NEU": "neutral",
    "SAD": "sadness",
}

# 12 fixed lexically-neutral sentences used for every CREMA-D recording.
# Source: README of the CheyneyComputerScience/CREMA-D repo.
SENTENCE_MAP: Dict[str, str] = {
    "IEO": "It's eleven o'clock.",
    "TIE": "That is exactly what happened.",
    "IOM": "I'm on my way to the meeting.",
    "IWW": "I wonder what this is about.",
    "TAI": "The airplane is almost full.",
    "MTI": "Maybe tomorrow it will be cold.",
    "IWL": "I would like a new alarm clock.",
    "ITH": "I think I have a doctor's appointment.",
    "DFA": "Don't forget a jacket.",
    "ITS": "I think I've seen this before.",
    "TSI": "The surface is slick.",
    "WSI": "We'll stop in a couple of minutes.",
}

INTENSITY_MAP: Dict[str, str] = {
    "LO": "low", "MD": "medium", "HI": "high", "XX": "unspecified",
}

# Re-use the canonical SmartHandover label space defined alongside MELD so
# downstream code does not need to know which dataset a row came from.
TARGET_LABELS = ["anger", "frustration", "sadness", "neutral", "satisfaction"]
TARGET_LABEL2ID = {label: i for i, label in enumerate(TARGET_LABELS)}

DEFAULT_ROOT = os.path.join("data", "raw", "CREMA-D")
DEFAULT_AUDIO_SUBDIR = "AudioWAV"
DEFAULT_DEMOGRAPHICS_CSV = "VideoDemographics.csv"


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------


def _parse_filename(name: str) -> Optional[Dict[str, str]]:
    """Parse ``1001_DFA_ANG_XX.wav`` into its 4 fields. Returns None on bad name."""
    base = os.path.splitext(os.path.basename(name))[0]
    parts = base.split("_")
    if len(parts) != 4:
        return None
    actor, sent, emo, intensity = parts
    return {
        "actor_id":    actor,
        "sentence_id": sent,
        "emotion_code": emo.upper(),
        "intensity_code": intensity.upper(),
    }


# ---------------------------------------------------------------------------
# Demographics (optional)
# ---------------------------------------------------------------------------


def _load_demographics(root: str) -> Dict[str, Dict[str, str]]:
    """Read VideoDemographics.csv if it exists. Maps actor_id -> dict."""
    path = os.path.join(root, DEFAULT_DEMOGRAPHICS_CSV)
    if not os.path.exists(path):
        return {}

    try:
        import pandas as pd
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  [WARN] Could not read demographics: {e}")
        return {}

    # The repo's CSV uses columns: ActorID, Age, Sex, Race, Ethnicity
    out: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        aid = str(row.get("ActorID", "")).strip()
        if not aid:
            continue
        out[aid] = {
            "age":    str(row.get("Age", "")),
            "sex":    str(row.get("Sex", "")),
            "race":   str(row.get("Race", "")),
            "ethnicity": str(row.get("Ethnicity", "")),
        }
    return out


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class CremadLoader:
    """Lightweight iterator over CREMA-D wav files with on-demand audio decoding.

    Acts a bit like a ``datasets.Dataset`` so that the rest of the project
    (which is built around the HuggingFace MELD loader) can consume it
    with minimal adaptation. Each yielded sample has the same keys as the
    MELD loader plus a few CREMA-D-specific extras::

        {
            "audio_id":         "1001_DFA_ANG_XX",
            "text":             "Don't forget a jacket.",
            "audio":            {"array": np.float32[N], "sampling_rate": 16000},
            "original_emotion": "ANG",
            "target_emotion":   "anger" | "satisfaction" | ... | "DROP",
            "target_label":     int    (or -1 for DROP rows)
            "intensity":        "low" | "medium" | "high" | "unspecified",
            "actor_id":         "1001",
            "sex":              "M" | "F" | "",   (only if demographics CSV present)
            "age":              "20" | ... | "",
            "race":             ...,
        }
    """

    def __init__(
        self,
        root: str = DEFAULT_ROOT,
        audio_subdir: str = DEFAULT_AUDIO_SUBDIR,
        label_map: Optional[Dict[str, Optional[str]]] = None,
        fear_as_frustration: bool = False,
        drop_unmapped: bool = True,
        sampling_rate: int = 16000,
    ):
        self.root = root
        self.audio_dir = os.path.join(root, audio_subdir)
        self.sampling_rate = sampling_rate
        self.drop_unmapped = drop_unmapped

        # Build the active label map - either the default or a user override,
        # plus the optional fear-as-frustration twist that mirrors MELD's
        # existing mapping choice.
        active = dict(label_map) if label_map else dict(CREMAD_LABEL_MAP_DEFAULT)
        if fear_as_frustration:
            active["FEA"] = "frustration"
        self.label_map = active

        if not os.path.isdir(self.audio_dir):
            raise FileNotFoundError(
                f"CREMA-D audio directory not found at: {self.audio_dir}\n"
                f"Place the CREMA-D wav files there. See "
                f"src/data/load_cremad.py module docstring for download "
                f"instructions."
            )

        # Index files once
        wavs = [f for f in os.listdir(self.audio_dir) if f.lower().endswith(".wav")]
        wavs.sort()

        self._records: List[Dict[str, str]] = []
        for fname in wavs:
            parsed = _parse_filename(fname)
            if parsed is None:
                continue
            target = self.label_map.get(parsed["emotion_code"])
            if target is None and self.drop_unmapped:
                continue
            self._records.append({**parsed, "filename": fname})

        # Demographics overlay (optional)
        self._demographics = _load_demographics(root)

    # -- HF Dataset-like API ------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx: int) -> Dict:
        rec = self._records[idx]
        emo_code = rec["emotion_code"]
        target = self.label_map.get(emo_code)

        sample = {
            "audio_id":         os.path.splitext(rec["filename"])[0],
            "text":             SENTENCE_MAP.get(rec["sentence_id"], ""),
            "audio":            self._load_audio(rec["filename"]),
            "original_emotion": emo_code,
            "target_emotion":   target if target is not None else "DROP",
            "target_label":     TARGET_LABEL2ID[target] if target else -1,
            "intensity":        INTENSITY_MAP.get(rec["intensity_code"], "unknown"),
            "actor_id":         rec["actor_id"],
            "sentence_id":      rec["sentence_id"],
        }

        # Overlay demographics if available
        demo = self._demographics.get(rec["actor_id"])
        if demo:
            sample.update(demo)

        return sample

    # -- audio decoding -----------------------------------------------------

    def _load_audio(self, filename: str) -> Dict:
        """Decode a single wav. Lazy import of soundfile to keep import cost low."""
        import soundfile as sf
        path = os.path.join(self.audio_dir, filename)
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if sr != self.sampling_rate:
            try:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sampling_rate)
                sr = self.sampling_rate
            except ImportError:
                # Last-resort linear interpolation
                new_len = int(round(len(wav) * self.sampling_rate / sr))
                wav = np.interp(
                    np.linspace(0, len(wav) - 1, new_len, dtype=np.float32),
                    np.arange(len(wav), dtype=np.float32),
                    wav,
                ).astype(np.float32)
                sr = self.sampling_rate
        return {"array": wav.astype(np.float32), "sampling_rate": int(sr)}


# ---------------------------------------------------------------------------
# Convenience: speaker-disjoint train/val/test split
# ---------------------------------------------------------------------------


def speaker_disjoint_split(
    loader: CremadLoader,
    val_actors: Iterable[str] = ("1051", "1052", "1053", "1054", "1055"),
    test_actors: Iterable[str] = ("1085", "1086", "1087", "1088", "1089",
                                  "1090", "1091"),
) -> Dict[str, List[int]]:
    """Return a dict {split: [indices]} where no actor appears in two splits.

    Default actor IDs are simply chunks at the end of the actor range; for
    serious experiments swap to a random seeded split with the actors as
    grouping keys (e.g. via ``GroupKFold``).
    """
    val_set = set(val_actors)
    test_set = set(test_actors)
    splits: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    for i, rec in enumerate(loader._records):
        aid = rec["actor_id"]
        if aid in test_set:
            splits["test"].append(i)
        elif aid in val_set:
            splits["val"].append(i)
        else:
            splits["train"].append(i)
    return splits


# ---------------------------------------------------------------------------
# Stats / sanity check
# ---------------------------------------------------------------------------


def print_stats(loader: CremadLoader) -> None:
    """Print class + intensity + speaker distributions to the console."""
    cls_counts = Counter()
    int_counts = Counter()
    speaker_counts = Counter()
    for rec in loader._records:
        emo = rec["emotion_code"]
        target = loader.label_map.get(emo)
        cls_counts[target if target is not None else "DROP"] += 1
        int_counts[INTENSITY_MAP.get(rec["intensity_code"], "unknown")] += 1
        speaker_counts[rec["actor_id"]] += 1

    total = sum(cls_counts.values())
    print("=" * 50)
    print(f"  CREMA-D | total kept rows: {total}")
    print(f"  Audio dir: {loader.audio_dir}")
    print("=" * 50)

    print("\n  Target class distribution:")
    for label in TARGET_LABELS + ["DROP"]:
        c = cls_counts.get(label, 0)
        if c == 0:
            continue
        pct = 100 * c / total
        bar = "#" * int(pct / 2)
        print(f"    {label:<14s} {c:>5d}  ({pct:5.1f}%)  {bar}")

    print("\n  Intensity distribution:")
    for k in ("low", "medium", "high", "unspecified"):
        c = int_counts.get(k, 0)
        if c == 0:
            continue
        print(f"    {k:<14s} {c:>5d}")

    print(f"\n  Distinct speakers kept: {len(speaker_counts)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 50)
    print("  SmartHandover - CREMA-D Dataset Loader")
    print("=" * 50)

    try:
        loader = CremadLoader()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    print_stats(loader)

    # Decode the first sample as a sanity check
    if len(loader):
        s = loader[0]
        print(f"\n--- Sample (idx 0) ---")
        print(f"  audio_id        : {s['audio_id']}")
        print(f"  text            : {s['text']}")
        print(f"  original / target: {s['original_emotion']} -> {s['target_emotion']}")
        print(f"  intensity       : {s['intensity']}")
        print(f"  actor_id        : {s['actor_id']}")
        print(f"  audio length    : "
              f"{len(s['audio']['array']) / s['audio']['sampling_rate']:.2f}s "
              f"@ {s['audio']['sampling_rate']} Hz")

    splits = speaker_disjoint_split(loader)
    print("\n  Default speaker-disjoint split sizes:")
    for k, v in splits.items():
        print(f"    {k:<6s} {len(v):>5d}")


if __name__ == "__main__":
    main()
