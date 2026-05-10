#!/usr/bin/env python3
"""SmartHandover - Day F13: Fine-tune wav2vec2 multi-corpus (Phase 7).

Replaces the frozen ``superb/wav2vec2-large-superb-er`` (W-F1 ~54% on
CREMA-D, P(sad)=0 on synthetic) with a checkpoint fine-tuned on a
multi-corpus mix.

Trains two variants for ablation (the master plan's Phase-7 §3.2):

  * ``no-synth``  : MELD train + CREMA-D train. Establishes the
                    ceiling reachable with real audio alone.
  * ``with-synth``: MELD train + CREMA-D train + synthetic phone-band
                    (sample weight 0.5 to keep the batch mostly real).

Both validate on **MELD val** only - never on synthetic, to avoid
overfit to TTS artefacts. Three test sets are reported per variant:
``meld_test``, ``cremad_test``, ``synth_test`` (cross-corpus eval).

Outputs
-------
* ``checkpoints/wav2vec2_finetuned_no_synth.pt``  (~1.2 GB)
* ``checkpoints/wav2vec2_finetuned.pt``           (~1.2 GB; with synth)
* ``data/processed/dayF13_results.csv``           (variant x test x metric)
* ``data/processed/dayF13_history_<variant>.json`` (loss/F1 per epoch)

Cost: zero (local GPU). Time: ~2-4 h total for both variants on the
RTX 5060 Ti, depending on early stopping.

Gate (per master plan)
----------------------
PASS: ``with-synth`` MELD test W-F1 >= frozen baseline (~44.8%)
      AND gap (synth_test_W-F1 - meld_test_W-F1) < 15 pp.
FAIL — gap > 15 pp: model learned TTS artefacts. Drop synth weight to
       0.25 and re-run, or fall back to ``no-synth``.
FAIL — MELD W-F1 below frozen: synthetic is corrupting the signal,
       use ``no-synth`` checkpoint and document as limitation.

Usage
-----
    python scripts/run_dayF13_finetune_wav2vec2.py                  # both variants
    python scripts/run_dayF13_finetune_wav2vec2.py --variant no-synth
    python scripts/run_dayF13_finetune_wav2vec2.py --variant with-synth
    python scripts/run_dayF13_finetune_wav2vec2.py --epochs 4       # quick run
    python scripts/run_dayF13_finetune_wav2vec2.py --batch-size 2   # less VRAM
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.train_audio import (  # noqa: E402
    AudioRecord, evaluate_checkpoint, load_cremad_audio_records,
    load_meld_audio_records, load_synth_phone_records, train_audio_model,
)


CKPT_DIR = "checkpoints"
PROCESSED_DIR = os.path.join("data", "processed")
RESULTS_CSV = os.path.join(PROCESSED_DIR, "dayF13_results.csv")
DEFAULT_VARIANTS = ("no-synth", "with-synth")


# ---------------------------------------------------------------------------


def build_test_records(meld_test, cremad_test, synth_test) -> Dict:
    return {
        "meld_test":   meld_test,
        "cremad_test": cremad_test,
        "synth_test":  synth_test,
    }


def _checkpoint_path(variant: str) -> str:
    if variant == "with-synth":
        return os.path.join(CKPT_DIR, "wav2vec2_finetuned.pt")
    return os.path.join(CKPT_DIR, "wav2vec2_finetuned_no_synth.pt")


# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=("no-synth", "with-synth", "both"),
                   default="both")
    p.add_argument("--epochs", type=int, default=8,
                   help="Max epochs per variant (default 8 + early stopping).")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Per-step batch (default 4; lower if OOM).")
    p.add_argument("--accumulation-steps", type=int, default=4,
                   help="Gradient accumulation; effective batch = "
                        "batch_size * accumulation_steps.")
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--synth-weight", type=float, default=0.5,
                   help="Sampler weight for synthetic clips (default 0.5).")
    p.add_argument("--limit-meld",   type=int, default=None,
                   help="Cap MELD records (testing).")
    p.add_argument("--limit-cremad", type=int, default=None)
    p.add_argument("--limit-synth",  type=int, default=None)
    args = p.parse_args()

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    variants = list(DEFAULT_VARIANTS) if args.variant == "both" else [args.variant]
    print("=" * 72)
    print("  Day F13 - Fine-tune wav2vec2 multi-corpus")
    print("=" * 72)
    print(f"  variants : {variants}")
    print(f"  epochs   : {args.epochs}")
    print(f"  batch    : {args.batch_size} (accumulation={args.accumulation_steps})")
    print(f"  synth wt : {args.synth_weight}")

    # ---- 1. Load all records once (shared across variants) ----------------
    print("\n[1/4] Loading audio records ...")
    print("  MELD train ...", end=" ", flush=True)
    meld_train = load_meld_audio_records(splits=("train",))
    if args.limit_meld:
        meld_train = meld_train[:args.limit_meld]
    print(f"{len(meld_train)} clips")

    print("  MELD val   ...", end=" ", flush=True)
    meld_val = load_meld_audio_records(splits=("validation",))
    print(f"{len(meld_val)} clips")

    print("  MELD test  ...", end=" ", flush=True)
    meld_test = load_meld_audio_records(splits=("test",))
    print(f"{len(meld_test)} clips")

    print("  CREMA-D train ...", end=" ", flush=True)
    cremad_train = load_cremad_audio_records(split="train")
    if args.limit_cremad:
        cremad_train = cremad_train[:args.limit_cremad]
    print(f"{len(cremad_train)} clips")

    print("  CREMA-D test  ...", end=" ", flush=True)
    cremad_test = load_cremad_audio_records(split="test")
    print(f"{len(cremad_test)} clips")

    print("  Synth phone   ...", end=" ", flush=True)
    synth_all = load_synth_phone_records(weight=args.synth_weight)
    if args.limit_synth:
        synth_all = synth_all[:args.limit_synth]
    # Carve a small synth_test (random 10%) for cross-corpus eval
    import random as _random
    rng = _random.Random(42)
    rng.shuffle(synth_all)
    n_synth_test = max(1, int(round(len(synth_all) * 0.10)))
    synth_test = synth_all[:n_synth_test]
    synth_train = synth_all[n_synth_test:]
    print(f"{len(synth_all)} (train {len(synth_train)}, test {len(synth_test)})")

    test_sets = build_test_records(meld_test, cremad_test, synth_test)

    # ---- 2. Train each variant -------------------------------------------
    rows: List[Dict] = []
    for variant in variants:
        print()
        print("=" * 72)
        print(f"  Variant: {variant}")
        print("=" * 72)

        if variant == "no-synth":
            train_records = list(meld_train) + list(cremad_train)
        else:  # with-synth
            train_records = (list(meld_train) + list(cremad_train)
                             + list(synth_train))

        ckpt_path = _checkpoint_path(variant)
        t0 = time.time()
        result = train_audio_model(
            train_records=train_records,
            val_records=meld_val,
            checkpoint_path=ckpt_path,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            accumulation_steps=args.accumulation_steps,
            patience=args.patience,
        )
        elapsed = time.time() - t0
        print(f"  variant '{variant}' trained in {elapsed/60:.1f} min")
        print(f"  best val W-F1 = {result['best_val_wf1']:.4f}")

        # Save history JSON
        hist_path = os.path.join(PROCESSED_DIR,
                                   f"dayF13_history_{variant}.json")
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump({"variant": variant, **result["history"]}, f, indent=2)
        print(f"  history -> {hist_path}")

        # ---- Evaluate on the 3 test sets --------------------------------
        print(f"\n  Evaluating {variant} on test sets ...")
        for test_name, test_records in test_sets.items():
            if not test_records:
                print(f"    (skip {test_name}: empty)")
                continue
            metrics = evaluate_checkpoint(ckpt_path, test_records,
                                            batch_size=args.batch_size * 2)
            rows.append({
                "variant":     variant,
                "test_set":    test_name,
                "n":           metrics["n"],
                "weighted_f1": metrics["weighted_f1"],
                "macro_f1":    metrics["macro_f1"],
                "frust_recall": metrics["frust_recall"],
                "loss":        metrics["loss"],
            })
            print(f"    {test_name:<12s} n={metrics['n']:<5d}  "
                  f"W-F1={metrics['weighted_f1']:.4f}  "
                  f"M-F1={metrics['macro_f1']:.4f}  "
                  f"FrustR={metrics['frust_recall']:.4f}")

    # ---- 3. Save results CSV --------------------------------------------
    if rows:
        with open(RESULTS_CSV, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Results CSV -> {RESULTS_CSV}")

    # ---- 4. Final pretty summary ----------------------------------------
    if rows:
        print()
        print("=" * 92)
        print("  Final summary (variant x test_set)")
        print("=" * 92)
        print(f"  {'variant':<12s} {'test_set':<12s} {'n':>6s} "
              f"{'W-F1':>8s} {'M-F1':>8s} {'FrustR':>8s}")
        print("  " + "-" * 60)
        for r in rows:
            print(f"  {r['variant']:<12s} {r['test_set']:<12s} {r['n']:>6d} "
                  f"{r['weighted_f1']:>8.4f} {r['macro_f1']:>8.4f} "
                  f"{r['frust_recall']:>8.4f}")

        # Gate verdict for the with-synth variant
        ws = [r for r in rows if r["variant"] == "with-synth"]
        if ws:
            meld = next((r for r in ws if r["test_set"] == "meld_test"), None)
            synth = next((r for r in ws if r["test_set"] == "synth_test"), None)
            if meld and synth:
                gap = synth["weighted_f1"] - meld["weighted_f1"]
                print()
                print("=" * 72)
                if meld["weighted_f1"] >= 0.448 and abs(gap) < 0.15:
                    print(f"  GATE: PASS  "
                          f"(meld_W-F1={meld['weighted_f1']:.3f}, "
                          f"gap={gap:+.3f})")
                elif gap > 0.15:
                    print(f"  GATE: FAIL (overfit synth) - "
                          f"gap={gap:+.3f} pp > 15 pp. Lower synth weight.")
                else:
                    print(f"  GATE: BORDERLINE - "
                          f"meld_W-F1={meld['weighted_f1']:.3f}; "
                          f"synth gap={gap:+.3f}")


if __name__ == "__main__":
    main()
