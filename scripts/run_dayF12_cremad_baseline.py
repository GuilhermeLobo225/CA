#!/usr/bin/env python3
"""SmartHandover - Day F12: Cross-corpus baseline of frozen wav2vec2 on CREMA-D.

The frozen ``superb/wav2vec2-large-superb-er`` was trained on IEMOCAP.
Running it on CREMA-D answers two questions for the report:

  1. **Generalisation.** Does the model carry IEMOCAP-trained knowledge
     across to a different actor pool / different recording set-up?
     The two corpora share the IEMOCAP-style 4-class space, but CREMA-D
     uses 91 actors and 12 fixed sentences (vs IEMOCAP's improvised
     dialogues), so any drop is informative.

  2. **Audio-component baseline.** This is the number to beat in
     Phase 7 (fine-tune). If frozen W-F1 on CREMA-D is, say, 60%,
     then a fine-tune that only reaches 62% has not added much; one
     that reaches 75% is where the synthetic + multi-corpus pipeline
     shows real value.

The 4 IEMOCAP classes map onto our 5 target labels as:

    ang -> anger
    hap -> satisfaction
    sad -> sadness
    neu -> neutral
    (frustration is not a CREMA-D class, by design.)

Output
------
* ``data/processed/cremad_speechbrain_predictions.csv``
   one row per clip with audio_id / actor_id / true_label / pred_label
   and the 4 IEMOCAP probability columns.
* ``data/processed/cremad_baseline_summary.json``
   with weighted/macro F1, accuracy and per-class precision/recall/F1.
* Console: classification report + confusion matrix.

Gate (per master plan)
----------------------
PASS: weighted F1 >= 50% on CREMA-D test split.
FAIL: < 35% means class mapping is wrong or audio path is broken.

Run
---
    python scripts/run_dayF12_cremad_baseline.py
    python scripts/run_dayF12_cremad_baseline.py --split test    # only 476 clips
    python scripts/run_dayF12_cremad_baseline.py --limit 50      # smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classifiers.speechbrain_classifier import SpeechBrainClassifier  # noqa: E402
from src.data.load_cremad import (  # noqa: E402
    CremadLoader,
    TARGET_LABELS,
    speaker_disjoint_split,
)
from src.evaluation.metrics import compute_metrics  # noqa: E402


# IEMOCAP -> our 5-class target mapping
IEMOCAP_TO_TARGET = {
    "ang": "anger",
    "hap": "satisfaction",
    "sad": "sadness",
    "neu": "neutral",
}

DEFAULT_OUT_CSV = os.path.join("data", "processed",
                                "cremad_speechbrain_predictions.csv")
DEFAULT_SUMMARY = os.path.join("data", "processed",
                                "cremad_baseline_summary.json")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Cross-corpus baseline: frozen wav2vec2 on CREMA-D",
    )
    p.add_argument("--audio-dir", default=None,
                   help="Override CREMA-D path (default: data/raw/CREMA-D).")
    p.add_argument("--split", choices=["all", "train", "val", "test"],
                   default="all",
                   help="Which split to evaluate (default: all 6171).")
    p.add_argument("--limit", type=int, default=None,
                   help="Smoke test: process only the first N clips.")
    p.add_argument("--output", default=DEFAULT_OUT_CSV)
    p.add_argument("--summary", default=DEFAULT_SUMMARY)
    args = p.parse_args()

    print("=" * 72)
    print("  Day F12 - Cross-corpus baseline: frozen wav2vec2 on CREMA-D")
    print("=" * 72)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device   : {device}")
    print(f"  split    : {args.split}")
    print(f"  output   : {args.output}")

    # --- 1. Load CREMA-D --------------------------------------------------
    loader_kwargs = {}
    if args.audio_dir:
        loader_kwargs["root"] = args.audio_dir
    loader = CremadLoader(**loader_kwargs)
    print(f"  total kept rows: {len(loader)}")

    if args.split == "all":
        indices = list(range(len(loader)))
    else:
        splits = speaker_disjoint_split(loader)
        indices = splits[args.split]
        print(f"  split '{args.split}': {len(indices)} clips")

    if args.limit is not None:
        indices = indices[:args.limit]
        print(f"  limit    : {len(indices)} (smoke test)")

    # --- 2. Load classifier ----------------------------------------------
    print(f"  loading  : superb/wav2vec2-large-superb-er ...")
    classifier = SpeechBrainClassifier(device=device)
    print()

    # --- 3. Iterate + predict --------------------------------------------
    rows: List[Dict] = []
    label_to_id = {lbl: i for i, lbl in enumerate(TARGET_LABELS)}

    t0 = time.time()
    skipped = 0
    for idx in tqdm(indices, desc="predict", unit="clip"):
        sample = loader[idx]
        audio = sample["audio"]["array"]
        sr = int(sample["audio"]["sampling_rate"])
        if audio is None or len(audio) == 0:
            skipped += 1
            continue

        try:
            probs = classifier.predict(audio.astype(np.float32), sr=sr)
        except Exception as e:
            print(f"\n  [WARN] predict failed for {sample['audio_id']}: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            skipped += 1
            continue

        # Map IEMOCAP probs onto our 5-class space (frustration stays at 0)
        target_probs = {lbl: 0.0 for lbl in TARGET_LABELS}
        for iemocap_lbl, target_lbl in IEMOCAP_TO_TARGET.items():
            target_probs[target_lbl] = float(probs.get(iemocap_lbl, 0.0))
        pred_label = max(target_probs, key=target_probs.get)

        rows.append({
            "audio_id":     sample["audio_id"],
            "actor_id":     sample.get("actor_id", ""),
            "intensity":    sample.get("intensity", ""),
            "true_label":   sample["target_emotion"],
            "true_id":      sample["target_label"],
            "pred_label":   pred_label,
            "pred_id":      label_to_id[pred_label],
            "p_ang":        round(probs.get("ang", 0.0), 4),
            "p_hap":        round(probs.get("hap", 0.0), 4),
            "p_sad":        round(probs.get("sad", 0.0), 4),
            "p_neu":        round(probs.get("neu", 0.0), 4),
        })

    elapsed = time.time() - t0

    if not rows:
        print("[ERROR] no clips evaluated.", file=sys.stderr)
        sys.exit(2)

    # --- 4. Save predictions ---------------------------------------------
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # --- 5. Metrics -------------------------------------------------------
    y_true = np.array([r["true_id"] for r in rows], dtype=np.int64)
    y_pred = np.array([r["pred_id"] for r in rows], dtype=np.int64)
    metrics = compute_metrics(y_true, y_pred, target_names=TARGET_LABELS)

    print()
    print("=" * 72)
    print(f"  Evaluated {len(rows)} clip(s) in {elapsed:.0f}s "
          f"({len(rows) / max(elapsed, 1):.1f} clips/s, skipped {skipped})")
    print("=" * 72)
    print(f"  accuracy     : {metrics['accuracy']:.4f}")
    print(f"  weighted F1  : {metrics['weighted_f1']:.4f}")
    print(f"  macro F1     : {metrics['macro_f1']:.4f}")
    print(f"  frust recall : {metrics['frustration_recall']:.4f}  "
          f"(n/a - CREMA-D has no frustration class)")

    print()
    print(f"  {'class':<14s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'n':>6s}")
    print("  " + "-" * 42)
    for lbl in TARGET_LABELS:
        d = metrics["per_class"].get(lbl, {})
        print(f"  {lbl:<14s} "
              f"{d.get('precision', 0.0):>6.3f} "
              f"{d.get('recall', 0.0):>6.3f} "
              f"{d.get('f1', 0.0):>6.3f} "
              f"{d.get('support', 0):>6d}")

    # --- 6. Confusion matrix ---------------------------------------------
    cm = metrics["confusion_matrix"]
    print()
    print("  Confusion matrix (rows=true, cols=pred):")
    short = [l[:5] for l in TARGET_LABELS]
    print("  " + " " * 8 + "".join(f"{s:>7s}" for s in short))
    for i, lbl in enumerate(TARGET_LABELS):
        row_str = "".join(f"{int(v):>7d}" for v in cm[i])
        print(f"  {short[i]:>7s} {row_str}")

    # --- 7. Persist summary JSON -----------------------------------------
    summary = {
        "split":            args.split,
        "n_clips":          len(rows),
        "skipped":          skipped,
        "elapsed_sec":      round(elapsed, 2),
        "weighted_f1":      metrics["weighted_f1"],
        "macro_f1":         metrics["macro_f1"],
        "accuracy":         metrics["accuracy"],
        "per_class":        metrics["per_class"],
        "iemocap_to_target": IEMOCAP_TO_TARGET,
    }
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # --- 8. Gate verdict --------------------------------------------------
    print()
    print("=" * 72)
    if metrics["weighted_f1"] >= 0.50:
        print(f"  GATE: PASS (W-F1={metrics['weighted_f1']:.3f} >= 0.50)")
        print("  Frozen wav2vec2 generalises adequately from IEMOCAP to "
              "CREMA-D.")
    elif metrics["weighted_f1"] >= 0.35:
        print(f"  GATE: BORDERLINE (W-F1={metrics['weighted_f1']:.3f}; "
              "0.35 <= x < 0.50)")
        print("  Generalisation is weak - fine-tune (Phase 7) should help.")
    else:
        print(f"  GATE: FAIL (W-F1={metrics['weighted_f1']:.3f} < 0.35)")
        print("  Mapping or audio path likely wrong. Inspect the confusion "
              "matrix.")

    print(f"\n  Predictions -> {args.output}")
    print(f"  Summary     -> {args.summary}")


if __name__ == "__main__":
    main()
