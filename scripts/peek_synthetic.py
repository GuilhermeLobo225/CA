#!/usr/bin/env python3
"""SmartHandover - Quick inspector for the synthetic text JSONL.

Run from project root:

    python scripts/peek_synthetic.py                       # 5 random samples
    python scripts/peek_synthetic.py --n 20                # 20 random samples
    python scripts/peek_synthetic.py --label frustration   # only that class
    python scripts/peek_synthetic.py --stats               # only the counts
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

# Force UTF-8 on stdout so curly quotes / em-dashes render correctly on
# Windows (default cp1252 turns them into '?'). Available since Python 3.7.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synthetic.text_normalize import normalise  # noqa: E402

DEFAULT_PATH = os.path.join("data", "synthetic", "text.jsonl")


# Legacy local mapping (no longer used - kept here so old smoke tests
# importing _PUNCT_FIXES from this module don't break).
_PUNCT_FIXES_LEGACY = {
    "‘": "'", "’": "'",   # left/right single quote
    "“": '"', "”": '"',   # left/right double quote
    "–": "-", "—": "-",   # en dash / em dash
    "…": "...",                # horizontal ellipsis
    " ": " ",                  # non-breaking space
}


def main() -> None:
    p = argparse.ArgumentParser(description="Peek at a synthetic-text JSONL.")
    p.add_argument("--path", default=DEFAULT_PATH)
    p.add_argument("--n", type=int, default=5,
                   help="How many random samples to print.")
    p.add_argument("--label", default=None,
                   choices=["anger", "frustration", "sadness", "neutral",
                            "satisfaction"],
                   help="Filter to one class.")
    p.add_argument("--stats", action="store_true",
                   help="Only print per-label counts.")
    p.add_argument("--seed", type=int, default=None,
                   help="Fix the random sample (default: fresh each call).")
    args = p.parse_args()

    if not os.path.exists(args.path):
        print(f"[ERROR] file not found: {args.path}")
        print("       Run generate_text first.")
        sys.exit(1)

    rows = []
    with open(args.path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"  file   : {args.path}")
    print(f"  total  : {len(rows)} samples")

    counts = Counter(r["label"] for r in rows)
    print("  by label:")
    for label in ("anger", "frustration", "sadness", "neutral", "satisfaction"):
        print(f"    {label:<14s} {counts.get(label, 0):>5d}")

    if args.stats:
        return

    if args.label:
        rows = [r for r in rows if r["label"] == args.label]
        print(f"  filtered to '{args.label}': {len(rows)} samples")

    if not rows:
        print("\n  Nothing to show.")
        return

    rng = random.Random(args.seed) if args.seed is not None else random
    n = min(args.n, len(rows))
    samples = rng.sample(rows, n)

    print()
    print("=" * 80)
    print(f"  {n} random sample(s)")
    print("=" * 80)
    for r in samples:
        text = normalise(r.get("text", ""))
        print()
        print(f"  id        : {r.get('id')}")
        print(f"  label     : {r.get('label')}  (intensity {r.get('intensity')}/5)")
        print(f"  style     : {r.get('style')}")
        print(f"  cause     : {r.get('cause')}")
        print(f"  persona   : {r.get('persona')}")
        print(f"  turn      : {r.get('turn_position')}")
        try:
            print(f"  text      : {text}")
        except UnicodeEncodeError:
            # Last-resort fallback if the console rejects even after fixes
            safe = text.encode("ascii", errors="replace").decode("ascii")
            print(f"  text      : {safe}")


if __name__ == "__main__":
    main()
