#!/usr/bin/env python3
"""SmartHandover - Day F10: Smoke eval with frozen wav2vec2.

Runs ``superb/wav2vec2-large-superb-er`` (the same model used in the
ensemble) on every clip in ``data/synthetic/smoke_audio/`` and reports
the IEMOCAP probability distribution per sample. The point of this
script is to answer one question: **is the synthetic audio carrying
emotional signal that wav2vec2 picks up?**

Output
------
* ``data/synthetic/smoke_frozen_eval.csv`` — one row per clip with:
  ``audio_id, voice, label_truth, p_ang, p_hap, p_sad, p_neu, predicted``
* Console summary: per-class median P(ang), confusion matrix.

Gate (per master plan)
----------------------
PASS:  classes ``frust`` and ``anger`` have median ``P(ang)`` > 0.25.
       (Anger sintético should activate the IEMOCAP "ang" class even
       though wav2vec2-IEMOCAP doesn't know "frustration" - frustration
       maps to ang as our designated proxy.)
FAIL — uniform ~0.10:
       TTS is producing emotionally flat audio. Inspect smoke_listening
       results; if humans also can't tell, suffix instructions are
       being ignored and we need stronger prompts (plan §3.2).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classifiers.speechbrain_classifier import SpeechBrainClassifier  # noqa: E402

DEFAULT_AUDIO_DIR = os.path.join("data", "synthetic", "smoke_audio")
DEFAULT_OUT_CSV   = os.path.join("data", "synthetic", "smoke_frozen_eval.csv")

# Mapping from our 5 target labels to the IEMOCAP-4 prediction we expect.
# Frustration is folded onto ``ang`` because wav2vec2-IEMOCAP doesn't
# have a frustration head and ``ang`` is our project-wide proxy.
LABEL_TO_IEMOCAP_TARGET = {
    "anger":        "ang",
    "frustration":  "ang",
    "sadness":      "sad",
    "neutral":      "neu",
    "satisfaction": "hap",
}


def _iter_wavs(audio_dir: str):
    """Yield (audio_id, voice, label, full_path) for every .wav under
    ``audio_dir/<label>/<id>_<voice>.wav``."""
    for label in sorted(os.listdir(audio_dir)):
        sub = os.path.join(audio_dir, label)
        if not os.path.isdir(sub):
            continue
        for fname in sorted(os.listdir(sub)):
            if not fname.lower().endswith(".wav"):
                continue
            base = os.path.splitext(fname)[0]
            # filename pattern: <id>_<voice>.wav
            if "_" in base:
                aid, voice = base.rsplit("_", 1)
            else:
                aid, voice = base, ""
            yield aid, voice, label, os.path.join(sub, fname)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR)
    p.add_argument("--output",    default=DEFAULT_OUT_CSV)
    args = p.parse_args()

    if not os.path.isdir(args.audio_dir):
        print(f"[ERROR] audio dir not found: {args.audio_dir}", file=sys.stderr)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device   : {device}")
    print(f"  audio    : {args.audio_dir}")
    print(f"  output   : {args.output}")
    print(f"  loading  : superb/wav2vec2-large-superb-er ...")

    classifier = SpeechBrainClassifier(device=device)
    print()

    rows: List[Dict] = []
    skipped = 0
    for aid, voice, label, path in _iter_wavs(args.audio_dir):
        try:
            wav, sr = sf.read(path, dtype="float32", always_2d=False)
        except Exception as e:
            print(f"  [WARN] could not read {path}: {e}", file=sys.stderr)
            skipped += 1
            continue
        if wav.ndim == 2:
            wav = wav.mean(axis=1)

        probs = classifier.predict(wav.astype(np.float32), sr=int(sr))
        # IEMOCAP labels: ang / hap / sad / neu
        predicted = max(probs, key=probs.get)
        rows.append({
            "audio_id":    aid,
            "voice":       voice,
            "label_truth": label,
            "expected_iemocap": LABEL_TO_IEMOCAP_TARGET.get(label, ""),
            "p_ang":       round(probs.get("ang", 0.0), 4),
            "p_hap":       round(probs.get("hap", 0.0), 4),
            "p_sad":       round(probs.get("sad", 0.0), 4),
            "p_neu":       round(probs.get("neu", 0.0), 4),
            "predicted":   predicted,
        })

    if not rows:
        print("[ERROR] no clips evaluated", file=sys.stderr)
        sys.exit(2)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # --- Summary ----------------------------------------------------------
    print()
    print("=" * 72)
    print(f"  Evaluated {len(rows)} clip(s) (skipped {skipped})")
    print("=" * 72)

    # Per-class median IEMOCAP probabilities
    by_label: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_label[r["label_truth"]].append(r)

    print()
    print(f"  {'label':<14s} {'n':>4s} "
          f"{'med_p_ang':>10s} {'med_p_hap':>10s} "
          f"{'med_p_sad':>10s} {'med_p_neu':>10s} "
          f"{'pred_match':>11s}")
    print("  " + "-" * 70)
    for label in sorted(by_label):
        items = by_label[label]
        med = lambda key: float(np.median([r[key] for r in items]))
        target = LABEL_TO_IEMOCAP_TARGET.get(label, "")
        n_match = sum(1 for r in items if r["predicted"] == target)
        match_pct = 100 * n_match / len(items) if items else 0.0
        print(f"  {label:<14s} {len(items):>4d} "
              f"{med('p_ang'):>10.3f} {med('p_hap'):>10.3f} "
              f"{med('p_sad'):>10.3f} {med('p_neu'):>10.3f} "
              f"{match_pct:>10.1f}%")

    # Headline gate: median P(ang) for anger and frustration
    anger_items = by_label.get("anger", [])
    frust_items = by_label.get("frustration", [])
    if anger_items:
        med_anger_p = float(np.median([r["p_ang"] for r in anger_items]))
        print(f"\n  median P(ang) for anger      = {med_anger_p:.3f}  "
              f"({'PASS' if med_anger_p > 0.25 else 'FAIL'} > 0.25)")
    if frust_items:
        med_frust_p = float(np.median([r["p_ang"] for r in frust_items]))
        print(f"  median P(ang) for frustration = {med_frust_p:.3f}  "
              f"({'PASS' if med_frust_p > 0.25 else 'FAIL'} > 0.25)")

    print(f"\n  CSV -> {args.output}")
    print("\nIf both medians are > 0.25, the audio carries emotional")
    print("signal that frozen wav2vec2 picks up. Proceed to Phase 4.")
    print("If they are ~0.10, the TTS is flat - revise instruction prompts")
    print("(see plan §3.2) and re-run smoke.")


if __name__ == "__main__":
    main()
