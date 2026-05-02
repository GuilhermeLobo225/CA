"""SmartHandover - Synthetic-data validation (listening test).

Picks a stratified random sample from the manifest, plays each clip and
asks the annotator (you) to pick a label blind. Stores answers and
finally computes Cohen's kappa against the LLM/TTS-claimed label,
plus per-class confusion matrix.

Three CLI subcommands::

    sample   : create / extend a sample sheet (CSV) of N clips.
    annotate : interactive listening test (uses the system audio player).
    score    : compute kappa, accuracy, confusion matrix from the sheet.

The sample sheet is CSV with columns:
    audio_id, label_truth, audio_path,
    annotator_<name>, ts_<name>

Multiple annotators can fill in their own column on the same sheet,
which is what you want for inter-rater agreement.

Usage
-----
    python -m src.data.synthetic.validate sample --n 200
    python -m src.data.synthetic.validate annotate --name guilherme
    python -m src.data.synthetic.validate annotate --name pedro
    python -m src.data.synthetic.validate score
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from src.data.synthetic import config as cfg

SHEET_PATH = os.path.join(cfg.OUT_DIR, "validation_sheet.csv")


# ---------------------------------------------------------------------------
# Sample creation (stratified)
# ---------------------------------------------------------------------------


def _read_manifest(path: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(row)
    return out


def _stratified_sample(manifest: List[Dict[str, str]], n: int,
                       seed: int) -> List[Dict[str, str]]:
    by_label: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in manifest:
        by_label[row["label"]].append(row)

    rng = random.Random(seed)
    per_class = max(1, n // max(len(by_label), 1))
    picked: List[Dict[str, str]] = []
    for label, rows in by_label.items():
        rng.shuffle(rows)
        picked.extend(rows[:per_class])

    # If we under-shot, top up with random from the remainder
    if len(picked) < n:
        rest = [r for r in manifest if r not in picked]
        rng.shuffle(rest)
        picked.extend(rest[:n - len(picked)])
    rng.shuffle(picked)
    return picked[:n]


def _load_or_create_sheet(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """Load existing sheet (rows + header order). Empty if missing."""
    if not os.path.exists(path):
        return [], []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
        header = reader.fieldnames or []
    return rows, header


def _save_sheet(path: str, rows: List[Dict[str, str]], header: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def cmd_sample(args: argparse.Namespace) -> None:
    manifest = _read_manifest(args.manifest)
    if not manifest:
        print(f"[ERROR] manifest empty or missing: {args.manifest}",
              file=sys.stderr)
        sys.exit(1)

    existing_rows, existing_header = _load_or_create_sheet(args.sheet)
    existing_ids = {r["audio_id"] for r in existing_rows}

    picks = _stratified_sample(manifest, args.n, args.seed)
    new_rows = []
    for p in picks:
        if p["audio_id"] in existing_ids:
            continue
        new_rows.append({
            "audio_id":    p["audio_id"],
            "label_truth": p["label"],
            "audio_path":  p["audio_path"],
        })

    if existing_header:
        header = existing_header
        # Make sure the 3 mandatory cols exist
        for col in ("audio_id", "label_truth", "audio_path"):
            if col not in header:
                header.insert(0, col)
    else:
        header = ["audio_id", "label_truth", "audio_path"]

    rows = existing_rows + new_rows
    _save_sheet(args.sheet, rows, header)

    print(f"  manifest    : {args.manifest}  ({len(manifest)} clips)")
    print(f"  sheet       : {args.sheet}  ({len(rows)} rows total)")
    print(f"  added now   : {len(new_rows)}")


# ---------------------------------------------------------------------------
# Annotation - interactive
# ---------------------------------------------------------------------------


def _play_audio(path: str) -> None:
    """Best-effort cross-platform playback. Returns immediately if no player."""
    if not os.path.exists(path):
        print(f"  [WARN] missing file: {path}", file=sys.stderr)
        return

    if sys.platform == "darwin":
        subprocess.run(["afplay", path], check=False)
    elif sys.platform == "win32":
        # Use PowerShell System.Media.SoundPlayer (handles wav natively)
        ps_cmd = (
            f"$p = New-Object System.Media.SoundPlayer '{path}'; "
            "$p.PlaySync();"
        )
        subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd],
                       check=False)
    else:
        # Linux - try common players in order
        for player in ("aplay", "paplay", "ffplay"):
            cmd = [player, path]
            if player == "ffplay":
                cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path]
            try:
                subprocess.run(cmd, check=False)
                return
            except FileNotFoundError:
                continue
        print("  [WARN] no audio player found (aplay/paplay/ffplay).",
              file=sys.stderr)


_LABEL_KEYS = {
    "1": "anger",
    "2": "frustration",
    "3": "sadness",
    "4": "neutral",
    "5": "satisfaction",
    "?": "uncertain",
    "s": "skip",
    "r": "replay",
    "q": "quit",
}


def _prompt_choice() -> str:
    print("    [1]anger  [2]frustration  [3]sadness  [4]neutral  "
          "[5]satisfaction  [?]unsure  [r]replay  [s]skip  [q]quit")
    while True:
        choice = input("    your label > ").strip().lower()
        if choice in _LABEL_KEYS:
            return choice
        print("    invalid. try again.")


def cmd_annotate(args: argparse.Namespace) -> None:
    rows, header = _load_or_create_sheet(args.sheet)
    if not rows:
        print(f"[ERROR] empty sheet at {args.sheet}. Run 'sample' first.",
              file=sys.stderr)
        sys.exit(1)

    ann_col = f"annotator_{args.name}"
    ts_col  = f"ts_{args.name}"
    if ann_col not in header:
        header.append(ann_col)
        header.append(ts_col)

    pending = [r for r in rows
               if not r.get(ann_col)]
    print(f"  sheet       : {args.sheet}")
    print(f"  annotator   : {args.name}")
    print(f"  total rows  : {len(rows)}")
    print(f"  pending     : {len(pending)}")
    if not pending:
        print("  Nothing to do for this annotator.")
        return

    print("\n  Listening test - blind labelling. Truth label is hidden.\n")

    done = 0
    for r in pending:
        path = r.get("audio_path", "")
        print(f"\n  [{done+1}/{len(pending)}] id={r['audio_id']}")
        print(f"    file: {path}")

        while True:
            _play_audio(path)
            choice_key = _prompt_choice()
            choice = _LABEL_KEYS[choice_key]
            if choice == "quit":
                _save_sheet(args.sheet, rows, header)
                print("  saved progress and exiting.")
                return
            if choice == "skip":
                break
            if choice == "replay":
                continue
            r[ann_col] = choice
            r[ts_col] = str(int(time.time()))
            done += 1
            # Save every 5 to be safe
            if done % 5 == 0:
                _save_sheet(args.sheet, rows, header)
            break

    _save_sheet(args.sheet, rows, header)
    print(f"\n  done. Annotated {done} new rows.")


# ---------------------------------------------------------------------------
# Scoring (Cohen's kappa, accuracy, confusion matrix)
# ---------------------------------------------------------------------------


def _cohens_kappa(y_true: List[str], y_pred: List[str],
                  labels: List[str]) -> float:
    """Compute Cohen's kappa for two label vectors of equal length."""
    if not y_true or len(y_true) != len(y_pred):
        return 0.0
    n = len(y_true)
    label_to_idx = {l: i for i, l in enumerate(labels)}

    cm = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t]][label_to_idx[p]] += 1

    po = sum(cm[i][i] for i in range(len(labels))) / n
    row_sums = [sum(row) for row in cm]
    col_sums = [sum(cm[r][c] for r in range(len(labels)))
                for c in range(len(labels))]
    pe = sum((row_sums[i] / n) * (col_sums[i] / n) for i in range(len(labels)))

    if 1 - pe == 0:
        return 0.0
    return (po - pe) / (1 - pe)


def _confusion_matrix(y_true: List[str], y_pred: List[str],
                      labels: List[str]) -> List[List[int]]:
    label_to_idx = {l: i for i, l in enumerate(labels)}
    cm = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t]][label_to_idx[p]] += 1
    return cm


def cmd_score(args: argparse.Namespace) -> None:
    rows, header = _load_or_create_sheet(args.sheet)
    if not rows:
        print(f"[ERROR] empty sheet at {args.sheet}.", file=sys.stderr)
        sys.exit(1)

    annotators = [c[len("annotator_"):]
                  for c in header if c.startswith("annotator_")]
    if not annotators:
        print("  No annotator columns yet. Run 'annotate' first.")
        return

    labels = cfg.TARGET_LABELS

    print("=" * 72)
    print("  Validation report")
    print("=" * 72)

    summary = {"annotators": {}, "inter_rater": {}}

    # Per-annotator agreement with truth
    for name in annotators:
        col = f"annotator_{name}"
        y_true = []
        y_pred = []
        for r in rows:
            t = r.get("label_truth", "")
            p = r.get(col, "")
            if t and p and p in labels:
                y_true.append(t)
                y_pred.append(p)
        if not y_true:
            print(f"\n  annotator {name:<14s} no annotations yet")
            continue

        n = len(y_true)
        acc = sum(1 for a, b in zip(y_true, y_pred) if a == b) / n
        kappa = _cohens_kappa(y_true, y_pred, labels)

        summary["annotators"][name] = {
            "n": n, "accuracy": acc, "kappa_vs_truth": kappa,
        }

        print(f"\n  {name:<14s}  n={n}  acc={acc:.3f}  "
              f"kappa(vs truth)={kappa:.3f}")

        cm = _confusion_matrix(y_true, y_pred, labels)
        print(f"    confusion (rows=truth, cols={name}):")
        print("      " + " ".join(f"{l[:5]:>7s}" for l in labels))
        for i, lbl in enumerate(labels):
            row_str = " ".join(f"{v:>7d}" for v in cm[i])
            print(f"      {lbl[:5]:>5s} {row_str}")

    # Inter-rater agreement (pairwise)
    if len(annotators) >= 2:
        print("\n  Inter-rater agreement (Cohen's kappa, pairwise):")
        for i in range(len(annotators)):
            for j in range(i + 1, len(annotators)):
                a, b = annotators[i], annotators[j]
                ca, cb = f"annotator_{a}", f"annotator_{b}"
                y_a = []
                y_b = []
                for r in rows:
                    pa = r.get(ca, "")
                    pb = r.get(cb, "")
                    if pa in labels and pb in labels:
                        y_a.append(pa)
                        y_b.append(pb)
                if not y_a:
                    print(f"    {a} vs {b}: no overlap yet")
                    continue
                k = _cohens_kappa(y_a, y_b, labels)
                summary["inter_rater"][f"{a}_vs_{b}"] = {
                    "n": len(y_a), "kappa": k,
                }
                print(f"    {a:<10s} vs {b:<10s}  n={len(y_a)}  kappa={k:.3f}")

    # Per-class recall/precision vs truth (averaged across annotators)
    if annotators:
        print("\n  Per-class recall (truth -> majority annotator vote):")
        per_class_correct: Counter = Counter()
        per_class_total: Counter = Counter()
        for r in rows:
            t = r.get("label_truth", "")
            if not t:
                continue
            votes = [r.get(f"annotator_{a}", "") for a in annotators]
            votes = [v for v in votes if v in labels]
            if not votes:
                continue
            mode = Counter(votes).most_common(1)[0][0]
            per_class_total[t] += 1
            if mode == t:
                per_class_correct[t] += 1

        for label in labels:
            tot = per_class_total.get(label, 0)
            ok = per_class_correct.get(label, 0)
            r = (ok / tot) if tot else 0.0
            print(f"    {label:<14s} recall={r:.3f}  ({ok}/{tot})")
        summary["majority_vs_truth"] = {
            label: {
                "n": per_class_total.get(label, 0),
                "recall": (per_class_correct.get(label, 0) /
                            per_class_total.get(label, 1))
                          if per_class_total.get(label) else 0.0,
            } for label in labels
        }

    # Persist
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    with open(cfg.VALIDATION_REPORT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Report saved -> {cfg.VALIDATION_REPORT}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description="Listening-test driver for synthetic-data validation."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_sample = sub.add_parser("sample",
                              help="Pick a stratified random subset to validate.")
    p_sample.add_argument("--manifest", default=cfg.MANIFEST_OUT)
    p_sample.add_argument("--sheet",    default=SHEET_PATH)
    p_sample.add_argument("--n", type=int, default=200)
    p_sample.add_argument("--seed", type=int, default=cfg.SEED)
    p_sample.set_defaults(func=cmd_sample)

    p_ann = sub.add_parser("annotate",
                           help="Interactive listening test for one annotator.")
    p_ann.add_argument("--sheet", default=SHEET_PATH)
    p_ann.add_argument("--name", required=True,
                       help="Annotator name (used as column suffix).")
    p_ann.set_defaults(func=cmd_annotate)

    p_sc = sub.add_parser("score",
                          help="Compute kappa / accuracy / confusion matrix.")
    p_sc.add_argument("--sheet", default=SHEET_PATH)
    p_sc.set_defaults(func=cmd_score)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
