#!/usr/bin/env python3
"""SmartHandover - Day F10: Smoke test TTS (50 samples, 3 voices).

Validates the TTS pipeline before spending ~€30 + 10h on the full
generation. Strategy:

  * Stratified sample of 50 utterances from ``text_filtered.jsonl``
    (10 per class).
  * 3 distinct voices in round-robin (default: nova, onyx, shimmer).
  * Per-sample emotional ``instructions`` built from the diversity axes
    (re-uses ``build_instruction`` from generate_audio).
  * Output: ``data/synthetic/smoke_audio/<label>/<id>_<voice>.wav``
            ``data/synthetic/smoke_manifest.csv``

Cost: ~€0.10-0.20 (depending on TTS model and clip length).

Usage
-----
    python scripts/run_dayF10_smoke_tts.py                 # default 50 samples
    python scripts/run_dayF10_smoke_tts.py --n-per-class 5 # 25 samples (cheaper)
    python scripts/run_dayF10_smoke_tts.py --resume        # skip already-generated

Gate (per master plan)
----------------------
PASS:  >=70% correct_class on the human listening test (after running
       validate.py annotate) AND classes frust/anger have median
       P(ang) > 0.25 on frozen wav2vec2 (after running smoke_eval).
FAIL — refusal mode: if >=10 samples produced empty/refused audio,
       soften the strongest instruction phrases (see plan §3.2).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synthetic import config as cfg  # noqa: E402
from src.data.synthetic._openai_client import (  # noqa: E402
    get_pool,
    load_dotenv,
    with_pool_backoff,
)
from src.data.synthetic.generate_audio import (  # noqa: E402
    _decode_wav_bytes,
    _save_wav,
    build_instruction,
)

# ---------------------------------------------------------------------------
# Constants (smoke-specific)
# ---------------------------------------------------------------------------

SMOKE_VOICES = ["nova", "verse", "shimmer"]    # 2 female + 1 male, all expressive
SMOKE_DIR     = os.path.join("data", "synthetic", "smoke_audio")
SMOKE_MANIFEST = os.path.join("data", "synthetic", "smoke_manifest.csv")

_voice_lock = threading.Lock()
_voice_counters: Dict[str, int] = {}


def _next_smoke_voice(label: str) -> str:
    """Round-robin between the 3 smoke voices, per class."""
    with _voice_lock:
        idx = _voice_counters.setdefault(label, 0)
        _voice_counters[label] += 1
    return SMOKE_VOICES[idx % len(SMOKE_VOICES)]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _read_filtered_jsonl(path: str) -> List[Dict]:
    out: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _stratified_sample(records: List[Dict], n_per_class: int,
                       seed: int = 42) -> List[Dict]:
    by_class: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_class[r["label"]].append(r)

    rng = random.Random(seed)
    out: List[Dict] = []
    for label in cfg.TARGET_LABELS:
        items = by_class.get(label, [])
        if not items:
            print(f"  [WARN] no records for class '{label}'", file=sys.stderr)
            continue
        rng.shuffle(items)
        if len(items) < n_per_class:
            print(f"  [WARN] only {len(items)} records for '{label}' "
                  f"(asked for {n_per_class})", file=sys.stderr)
        out.extend(items[:n_per_class])
    return out


# ---------------------------------------------------------------------------
# Per-sample TTS (single-call, robust)
# ---------------------------------------------------------------------------


def _generate_one_wav(record: Dict, voice: str, audio_root: str,
                       model: str) -> Optional[Dict]:
    """Generate one .wav for ``record`` with the given voice. Returns a
    manifest row dict, or None on failure."""
    label = record["label"]
    sample_id = record["id"]
    text = record["text"]
    instruction = build_instruction(record)

    out_dir = os.path.join(audio_root, label)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{sample_id}_{voice}.wav")

    pool = get_pool("tts")

    def _call(client):
        return client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            instructions=instruction,
            response_format="wav",
        )

    try:
        resp = with_pool_backoff(pool, _call)
    except Exception as e:
        print(f"\n  [WARN] TTS failed for {sample_id} ({voice}): "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return None

    # Read bytes
    audio_bytes: Optional[bytes] = None
    try:
        audio_bytes = resp.read()
    except Exception:
        try:
            buf = io.BytesIO()
            resp.stream_to_file(buf)
            audio_bytes = buf.getvalue()
        except Exception:
            audio_bytes = getattr(resp, "content", None)

    if not audio_bytes:
        print(f"\n  [WARN] empty audio for {sample_id} ({voice})",
              file=sys.stderr)
        return None

    try:
        wav, sr = _decode_wav_bytes(audio_bytes)
    except Exception as e:
        print(f"\n  [WARN] decode failed for {sample_id}: {e}",
              file=sys.stderr)
        return None

    # Resample to 16 kHz so the frozen wav2vec2 can consume directly.
    target_sr = 16000
    if sr != target_sr:
        try:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except ImportError:
            pass

    _save_wav(out_path, wav, sr)
    duration = float(len(wav) / sr)

    return {
        "audio_id":     sample_id,
        "voice":        voice,
        "label":        label,
        "label_id":     record["label_id"],
        "text":         text,
        "intensity":    record["intensity"],
        "cause":        record["cause"],
        "style":        record["style"],
        "persona":      record["persona"],
        "turn_position": record["turn_position"],
        "instruction":  instruction,
        "duration_sec": round(duration, 3),
        "sample_rate":  int(sr),
        "model":        model,
        "audio_path":   out_path.replace("\\", "/"),
        "ts":           time.time(),
    }


_MANIFEST_FIELDS = [
    "audio_id", "voice", "label", "label_id", "text",
    "intensity", "cause", "style", "persona", "turn_position",
    "instruction", "duration_sec", "sample_rate", "model",
    "audio_path", "ts",
]


_manifest_lock = threading.Lock()


def _append_manifest_row(path: str, row: Dict) -> None:
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    with _manifest_lock:
        with open(path, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_MANIFEST_FIELDS,
                                extrasaction="ignore")
            if is_new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in _MANIFEST_FIELDS})


def _load_done_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    seen = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            aid = row.get("audio_id")
            voice = row.get("voice")
            if aid and voice:
                seen.add(f"{aid}_{voice}")
    return seen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test TTS - 50 samples / 3 voices.")
    p.add_argument("--input", default=cfg.TEXT_FILTERED,
                   help="Filtered synthetic JSONL (default: text_filtered.jsonl)")
    p.add_argument("--audio-root", default=SMOKE_DIR)
    p.add_argument("--manifest", default=SMOKE_MANIFEST)
    p.add_argument("--n-per-class", type=int, default=10,
                   help="Samples per class (default 10 = 50 total).")
    p.add_argument("--workers", type=int, default=cfg.AUDIO_CONCURRENCY)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-resume", action="store_true",
                   help="Truncate manifest and re-generate everything.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()

    print("=" * 72)
    print("  SmartHandover - Day F10: Smoke TTS")
    print("=" * 72)
    print(f"  input        : {args.input}")
    print(f"  audio root   : {args.audio_root}")
    print(f"  manifest     : {args.manifest}")
    print(f"  voices       : {SMOKE_VOICES}")
    print(f"  n per class  : {args.n_per_class}  (total = {args.n_per_class * 5})")
    print(f"  workers      : {args.workers}")
    print(f"  TTS model    : {cfg.TTS_MODEL}")

    if not os.path.exists(args.input):
        print(f"[ERROR] input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # --- Sampling ----------------------------------------------------------
    records = _read_filtered_jsonl(args.input)
    print(f"  records read : {len(records)}")
    sampled = _stratified_sample(records, args.n_per_class, seed=args.seed)
    print(f"  sampled      : {len(sampled)}")

    # Assign voice deterministically per (label, position-within-class)
    work_items: List[tuple] = []  # (record, voice, key)
    counters: Dict[str, int] = {}
    for rec in sampled:
        i = counters.setdefault(rec["label"], 0)
        counters[rec["label"]] += 1
        voice = SMOKE_VOICES[i % len(SMOKE_VOICES)]
        work_items.append((rec, voice, f"{rec['id']}_{voice}"))

    # --- Resume ------------------------------------------------------------
    if args.no_resume:
        if os.path.exists(args.manifest):
            os.remove(args.manifest)
        already = set()
    else:
        already = _load_done_ids(args.manifest)
    if already:
        print(f"  resuming     : {len(already)} already done")
        work_items = [w for w in work_items if w[2] not in already]

    if not work_items:
        print("\n  Nothing to do.")
        return

    # --- TTS pool eager check ---------------------------------------------
    try:
        get_pool("tts")
    except RuntimeError as e:
        print(f"\n[ERROR] TTS pool not configured: {e}", file=sys.stderr)
        sys.exit(2)

    # --- Run ---------------------------------------------------------------
    t0 = time.time()
    n_ok, n_fail = 0, 0
    durations: List[float] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_generate_one_wav, rec, voice, args.audio_root,
                        cfg.TTS_MODEL): (rec, voice)
            for (rec, voice, _key) in work_items
        }
        with tqdm(total=len(futures), desc="tts", unit="clip") as bar:
            for fut in as_completed(futures):
                row = fut.result()
                if row is None:
                    n_fail += 1
                else:
                    _append_manifest_row(args.manifest, row)
                    durations.append(row["duration_sec"])
                    n_ok += 1
                bar.set_postfix(ok=n_ok, fail=n_fail)
                bar.update(1)

    elapsed = time.time() - t0
    avg = sum(durations) / len(durations) if durations else 0.0

    print()
    print(f"  ok={n_ok}  fail={n_fail}  elapsed={elapsed:.0f}s "
          f"({n_ok / max(elapsed, 1):.2f} clips/s)")
    print(f"  avg clip duration: {avg:.2f}s")
    print(f"  manifest -> {args.manifest}")
    print(f"  audio    -> {args.audio_root}/<label>/<id>_<voice>.wav")
    print()
    print("Next:")
    print("  1. Listening test (manual, ~30 min):")
    print(f"     python -m src.data.synthetic.validate sample "
          f"--manifest {args.manifest} "
          f"--sheet data/synthetic/smoke_listening.csv --n 50")
    print(f"     python -m src.data.synthetic.validate annotate "
          f"--sheet data/synthetic/smoke_listening.csv --name <yourname>")
    print("  2. Frozen wav2vec2 eval (auto):")
    print(f"     python scripts/run_dayF10_smoke_eval.py "
          f"--audio-dir {args.audio_root}")


if __name__ == "__main__":
    main()
