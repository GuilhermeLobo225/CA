#!/usr/bin/env python3
"""SmartHandover - Day F9: re-build features + train meta-classifier v2.

Uses the new RoBERTa checkpoint (``roberta_combined.pt`` from Day F8) to:

  1. Regenerate per-split RoBERTa predictions on the MELD splits and
     cache them as ``roberta_*_predictions_v2.csv``.
  2. Build new 19-dim ensemble features (only the RoBERTa columns
     change; VADER, GoEmotions, SpeechBrain stay frozen).
  3. Train and select the best of LogisticRegression / XGBoost / MLP.
     Saved to ``checkpoints/meta_classifier_v2.pkl``.
  4. Re-run the 3-way fusion comparison (score / late / decision)
     against the new features. Output:
       - ``data/processed/fusion_comparison_v2.csv``
       - ``data/processed/fusion_comparison_v2.png``
       - ``data/processed/fusion_late_weights_v2.json``

Gate (per the master plan)
--------------------------
PASS:  ensemble (any of the 3 fusion strategies) reaches W-F1 >= 68%
       AND frust recall >= 45% on MELD test.
FAIL:  if W-F1 drops below the current 66.0%, the new RoBERTa hurt the
       ensemble. Investigate weights or stick with v1.

Usage
-----
    python scripts/run_dayF9_meta_v2.py                            # default
    python scripts/run_dayF9_meta_v2.py --roberta-checkpoint <path>
    python scripts/run_dayF9_meta_v2.py --skip-meta                # only fusion
    python scripts/run_dayF9_meta_v2.py --skip-fusion              # only meta
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_CHECKPOINT = os.path.join("checkpoints", "roberta_combined.pt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roberta-checkpoint", default=DEFAULT_CHECKPOINT,
                   help=f"Path to the new RoBERTa .pt (default: {DEFAULT_CHECKPOINT})")
    p.add_argument("--suffix", default="_v2",
                   help="Suffix appended to all v2 outputs (default: '_v2')")
    p.add_argument("--skip-meta", action="store_true",
                   help="Skip the meta-classifier training (only run fusion).")
    p.add_argument("--skip-fusion", action="store_true",
                   help="Skip the fusion comparison (only train meta).")
    p.add_argument("--no-regenerate", action="store_true",
                   help="Don't force regeneration of RoBERTa CSVs if cache "
                        "files for this suffix already exist.")
    return p.parse_args()


def run(cmd: list, label: str) -> None:
    print()
    print("=" * 72)
    print(f"  {label}")
    print(f"  command: {' '.join(cmd)}")
    print("=" * 72)
    t0 = time.time()
    rc = subprocess.call(cmd)
    dt = time.time() - t0
    if rc != 0:
        print(f"\n[FAIL] {label} returned exit code {rc}", file=sys.stderr)
        sys.exit(rc)
    print(f"\n[OK] {label} done in {dt:.0f}s")


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.roberta_checkpoint):
        print(f"[ERROR] checkpoint not found: {args.roberta_checkpoint}",
              file=sys.stderr)
        sys.exit(2)

    print("SmartHandover - Day F9: meta-classifier v2 + fusion v2")
    print(f"  RoBERTa checkpoint : {args.roberta_checkpoint}")
    print(f"  Suffix             : {args.suffix}")

    if not args.skip_meta:
        # Step 1+2+3: regenerate predictions, build features, train meta.
        cmd = [
            sys.executable, "-m", "src.classifiers.ensemble_trainer",
            "--roberta-checkpoint", args.roberta_checkpoint,
            "--output-suffix", args.suffix,
        ]
        if not args.no_regenerate:
            cmd.append("--force-regenerate")
        run(cmd, "Step A - regenerate features + train meta-classifier v2")

    if not args.skip_fusion:
        # Step 4: 3-way fusion comparison on the v2 features.
        cmd = [
            sys.executable, "-m", "src.classifiers.fusion_strategies",
            "--features-suffix", args.suffix,
            "--output-suffix",   args.suffix,
        ]
        run(cmd, "Step B - fusion comparison (score / late / decision)")

    # Final summary path printout
    print()
    print("=" * 72)
    print("  Day F9 outputs")
    print("=" * 72)
    suffix = args.suffix
    paths = [
        f"data/processed/roberta_train_predictions{suffix}.csv",
        f"data/processed/roberta_val_predictions{suffix}.csv",
        f"data/processed/roberta_predictions{suffix}.csv",
        f"data/processed/ensemble_features_train{suffix}.csv",
        f"data/processed/ensemble_features_val{suffix}.csv",
        f"data/processed/ensemble_features_test{suffix}.csv",
        f"checkpoints/meta_classifier{suffix}.pkl",
        f"data/processed/meta_classifier{suffix}_summary.json",
        f"data/processed/fusion_comparison{suffix}.csv",
        f"data/processed/fusion_comparison{suffix}.png",
        f"data/processed/fusion_late_weights{suffix}.json",
    ]
    for p in paths:
        marker = "OK" if os.path.exists(p) else "missing"
        print(f"  [{marker:>7s}] {p}")

    print()
    print("Inspect data/processed/meta_classifier_v2_summary.json for the")
    print("test metrics by classifier and the chosen 'best_meta_classifier'.")
    print("Inspect data/processed/fusion_comparison_v2.csv for the 3-way")
    print("comparison (score / late / decision).")
    print()
    print("Gate: PASS if any fusion strategy reaches W-F1 >= 68% AND")
    print("      frust recall >= 45% on MELD test.")


if __name__ == "__main__":
    main()
