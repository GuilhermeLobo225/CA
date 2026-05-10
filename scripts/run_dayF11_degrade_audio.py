#!/usr/bin/env python3
"""SmartHandover - Day F11: Telephone-band degradation of synthetic audio.

Reads every ``.wav`` under ``data/synthetic/audio/<label>/`` (the clean
24 kHz output of ``generate_audio.py``), applies the standard
``degrade_pipeline`` (mu-law + bandpass + noise + gain jitter), and
writes the degraded ``audio_phone/<label>/<id>.wav`` (16 kHz mono).

The clean and degraded versions are kept in parallel so:

  * **Listening test** (Phase 8) can use ``audio/`` (cleanest signal).
  * **Fine-tune wav2vec2** (Phase 7) can use ``audio_phone/``
    (realistic signal).

Cost: zero. Pure local processing.

Usage
-----
    python scripts/run_dayF11_degrade_audio.py              # default settings
    python scripts/run_dayF11_degrade_audio.py --noise-dir data/raw/musan/noise/free-sound
    python scripts/run_dayF11_degrade_audio.py --snr-low 18 --snr-high 28
    python scripts/run_dayF11_degrade_audio.py --workers 8
    python scripts/run_dayF11_degrade_audio.py --limit 10   # smoke test (10 clips)
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from src.data.synthetic import config as cfg  # noqa: E402
from src.data.synthetic.degrade_audio import degrade_pipeline  # noqa: E402

DEFAULT_INPUT_DIR  = os.path.join("data", "synthetic", "audio")
DEFAULT_OUTPUT_DIR = os.path.join("data", "synthetic", "audio_phone")
DEFAULT_MANIFEST   = os.path.join("data", "synthetic", "audio_phone_manifest.csv")

_MANIFEST_FIELDS = [
    "audio_id", "label", "src_path", "dst_path",
    "duration_sec", "sample_rate",
    "snr_db", "noise_source", "codec", "bandpass_hz",
    "ts",
]

# ---------------------------------------------------------------------------


def _iter_clean_wavs(input_dir: str):
    """Yield (label, audio_id, full_path) for each clean .wav."""
    for label in sorted(os.listdir(input_dir)):
        sub = os.path.join(input_dir, label)
        if not os.path.isdir(sub):
            continue
        for fname in sorted(os.listdir(sub)):
            if not fname.lower().endswith(".wav"):
                continue
            audio_id = os.path.splitext(fname)[0]
            yield label, audio_id, os.path.join(sub, fname)


def _load_done_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    seen = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            aid = row.get("audio_id")
            if aid:
                seen.add(aid)
    return seen


_manifest_lock = None  # filled in main()


def _append_manifest(path: str, row: Dict, lock) -> None:
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    with lock:
        with open(path, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_MANIFEST_FIELDS,
                                extrasaction="ignore")
            if is_new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in _MANIFEST_FIELDS})


def _process_one(label: str, audio_id: str, src_path: str,
                 output_dir: str, snr_lo: float, snr_hi: float,
                 noise_dir: Optional[str], seed: int) -> Optional[Dict]:
    """Degrade one clip. Returns manifest row or None on failure."""
    out_dir = os.path.join(output_dir, label)
    os.makedirs(out_dir, exist_ok=True)
    dst_path = os.path.join(out_dir, f"{audio_id}.wav")

    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        # Already done - just refresh manifest line
        try:
            wav, sr = sf.read(dst_path, dtype="float32", always_2d=False)
            duration = float(len(wav) / sr) if sr else 0.0
        except Exception:
            duration = 0.0
            sr = 16000
        return {
            "audio_id":   audio_id,
            "label":      label,
            "src_path":   src_path.replace("\\", "/"),
            "dst_path":   dst_path.replace("\\", "/"),
            "duration_sec": round(duration, 3),
            "sample_rate": int(sr),
            "snr_db":     "(cached)",
            "noise_source": "(cached)",
            "codec":      "mu_law_8k",
            "bandpass_hz": "300-3400",
            "ts":         time.time(),
        }

    try:
        wav, sr = sf.read(src_path, dtype="float32", always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)

        # Per-clip seed = stable hash of audio_id, so re-runs match
        clip_seed = (seed + abs(hash(audio_id))) & 0x7FFFFFFF
        wav_phone, sr_out, meta = degrade_pipeline(
            wav.astype(np.float32),
            sr_in=int(sr),
            snr_db_range=(snr_lo, snr_hi),
            noise_dir=noise_dir,
            seed=clip_seed,
        )
    except Exception as e:
        print(f"\n  [WARN] degrade failed for {audio_id}: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return None

    sf.write(dst_path, wav_phone, sr_out, subtype="PCM_16")
    duration = float(len(wav_phone) / sr_out)
    return {
        "audio_id":     audio_id,
        "label":        label,
        "src_path":     src_path.replace("\\", "/"),
        "dst_path":     dst_path.replace("\\", "/"),
        "duration_sec": round(duration, 3),
        "sample_rate":  int(sr_out),
        "snr_db":       meta["snr_db"],
        "noise_source": meta["noise_source"],
        "codec":        meta["codec"],
        "bandpass_hz":  meta["bandpass_hz"],
        "ts":           time.time(),
    }


# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Telephone-band degradation of synthetic audio."
    )
    p.add_argument("--input-dir",  default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--manifest",   default=DEFAULT_MANIFEST)
    p.add_argument("--snr-low",  type=float, default=15.0,
                   help="Lower bound of random SNR in dB (default 15).")
    p.add_argument("--snr-high", type=float, default=25.0,
                   help="Upper bound of random SNR in dB (default 25).")
    p.add_argument("--noise-dir", default=None,
                   help="Directory of recorded noise .wav (e.g. MUSAN). "
                        "If unset, uses synthetic pink+hum noise.")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel worker threads (default 4).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N clips (smoke test).")
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    print("=" * 72)
    print("  SmartHandover - Day F11: Phone-band degradation")
    print("=" * 72)
    print(f"  input  : {args.input_dir}")
    print(f"  output : {args.output_dir}")
    print(f"  manifest: {args.manifest}")
    print(f"  SNR    : [{args.snr_low}, {args.snr_high}] dB")
    print(f"  noise  : {args.noise_dir or '(synthetic pink+hum)'}")
    print(f"  workers: {args.workers}")
    print(f"  seed   : {args.seed}")

    if not os.path.isdir(args.input_dir):
        print(f"[ERROR] input dir not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    # --- Index clean wavs --------------------------------------------------
    work: List[tuple] = list(_iter_clean_wavs(args.input_dir))
    print(f"  total clean wavs: {len(work)}")

    if args.no_resume and os.path.exists(args.manifest):
        os.remove(args.manifest)

    if not args.no_resume:
        already = _load_done_ids(args.manifest)
        if already:
            print(f"  resuming - skipping {len(already)} already-degraded ids")
            work = [w for w in work if w[1] not in already]

    if args.limit is not None:
        work = work[:args.limit]
        print(f"  limit  : {len(work)}")

    if not work:
        print("\n  Nothing to do.")
        return

    import threading
    lock = threading.Lock()

    t0 = time.time()
    n_ok, n_fail = 0, 0
    durations: List[float] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_process_one, label, aid, src,
                        args.output_dir, args.snr_low, args.snr_high,
                        args.noise_dir, args.seed):
            (label, aid, src)
            for (label, aid, src) in work
        }
        with tqdm(total=len(futures), desc="degrade", unit="clip") as bar:
            for fut in as_completed(futures):
                row = fut.result()
                if row is None:
                    n_fail += 1
                else:
                    _append_manifest(args.manifest, row, lock)
                    durations.append(row["duration_sec"])
                    n_ok += 1
                bar.set_postfix(ok=n_ok, fail=n_fail)
                bar.update(1)

    elapsed = time.time() - t0
    print()
    print(f"  ok={n_ok}  fail={n_fail}  elapsed={elapsed:.0f}s "
          f"({n_ok / max(elapsed, 1):.1f} clips/s)")
    if durations:
        print(f"  avg duration: {sum(durations) / len(durations):.2f}s")
    print(f"  manifest -> {args.manifest}")
    print(f"  audio    -> {args.output_dir}/<label>/<id>.wav")

    if n_fail:
        print("\n  Some clips failed - re-run to retry (resume kicks in).")


if __name__ == "__main__":
    main()
