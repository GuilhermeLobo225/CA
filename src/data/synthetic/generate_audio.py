"""SmartHandover - Synthetic audio generator (gpt-4o-mini-tts).

Reads ``text_filtered.jsonl`` and produces one .wav per record using
OpenAI's ``gpt-4o-mini-tts``. The TTS instruction is built per-sample
from the diversity axes already in the record, so prosody varies along
with the text. Voices are round-robin across all available IDs to avoid
speaker-identity bias in the audio classifier.

Output layout
-------------
    data/synthetic/
      audio/
        anger/        anger_00001.wav
        frustration/  frust_00042.wav
        ...
      manifest.csv    # one row per generated wav

Manifest schema (CSV)::

    audio_id, label, label_id, text,
    intensity, cause, style, persona, turn_position,
    voice, instruction_hash, duration_sec, sample_rate,
    judge_score, judge_intensity_obs, judge_natural,
    model, ts, audio_path

Usage
-----
    python -m src.data.synthetic.generate_audio                  # full run
    python -m src.data.synthetic.generate_audio --limit 10       # smoke test
    python -m src.data.synthetic.generate_audio --workers 2      # gentler rate
    python -m src.data.synthetic.generate_audio --label frustration --limit 100
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from tqdm import tqdm

from src.data.synthetic import config as cfg
from src.data.synthetic._openai_client import (
    get_pool,
    load_dotenv,
    with_pool_backoff,
)

# ---------------------------------------------------------------------------
# Voice round-robin (deterministic, label-aware)
# ---------------------------------------------------------------------------


_voice_lock = threading.Lock()
_voice_counters: Dict[str, int] = {label: 0 for label in cfg.TARGET_LABELS}


def _next_voice(label: str) -> str:
    """Deterministic round-robin per class. Each class progresses
    independently, so within a single class voices are evenly distributed."""
    with _voice_lock:
        idx = _voice_counters[label] % len(cfg.TTS_VOICES)
        _voice_counters[label] += 1
        return cfg.TTS_VOICES[idx]


# ---------------------------------------------------------------------------
# Instruction templates per label
# ---------------------------------------------------------------------------


_INTENSITY_TONE = {
    1: "calm and even",
    2: "slightly elevated",
    3: "noticeably tense",
    4: "agitated and forceful",
    5: "very intense, on the edge of losing composure",
}

_LABEL_BASE = {
    "anger": (
        "Speak as an angry customer on a phone call to support. The voice "
        "should sound assertive, confrontational, and clearly upset."
    ),
    "frustration": (
        "Speak as a frustrated customer who has been struggling with the "
        "same issue. The voice should convey weariness, repeated effort "
        "wasted, and growing impatience."
    ),
    "sadness": (
        "Speak as a customer who feels disappointed and let down. The "
        "voice should sound subdued, slightly hollow, with a touch of "
        "resignation."
    ),
    "neutral": (
        "Speak as a customer making a routine, non-emotional remark. "
        "The voice should be matter-of-fact, neither warm nor cold."
    ),
    "satisfaction": (
        "Speak as a satisfied customer whose issue has been handled "
        "well. The voice should sound warm, relieved, and grateful."
    ),
}

_STYLE_HINTS = {
    "explicit_anger":     "raise volume on emphatic words, tight clipped pacing",
    "passive_aggressive": "even surface tone with sarcastic emphasis on key words",
    "sarcastic":          "drawn-out sarcastic intonation, eye-rolling implied",
    "exasperated":        "audible sigh-like quality, tired delivery",
    "tearful":            "voice on the edge of tears, slightly wavering",
    "polite_but_firm":    "controlled and measured, but with steel underneath",
    "rambling":           "slightly disorganised, occasional self-correction",
    "matter_of_fact":     "neutral pacing, factual",
    "polite_formal":      "courteous register, careful articulation",
    "casual":             "informal and relaxed",
    "grateful":           "warm, slightly relieved",
    "warm":               "friendly and open",
}

_PERSONA_HINTS = {
    "young_informal":      "younger speaker, contemporary informal English",
    "elderly_formal":      "older speaker, slightly slower pace, formal phrasing",
    "business_user":       "professional register, time-conscious",
    "technically_savvy":   "comfortable with technical terms",
    "first_time_caller":   "uncertain, maybe a bit nervous",
    "regular_complainant": "knows the routine, less patient with formalities",
}

_TURN_HINTS = {
    "opening":    "the call has just started",
    "mid_call":   "the call is in progress",
    "escalation": "this turn is a clear escalation point",
    "closing":    "this is near the end of the call",
}


def build_instruction(record: Dict) -> str:
    label = record["label"]
    intensity = int(record["intensity"])
    style = record["style"]
    persona = record["persona"]
    turn = record["turn_position"]

    parts = [
        _LABEL_BASE.get(label, ""),
        f"Intensity: {_INTENSITY_TONE.get(intensity, 'moderate')} "
        f"({intensity}/5).",
        f"Style: {_STYLE_HINTS.get(style, style)}.",
        f"Persona: {_PERSONA_HINTS.get(persona, persona)}.",
        f"Turn position: {_TURN_HINTS.get(turn, turn)}.",
        "Add small naturalistic disfluencies (a brief breath, an "
        "occasional micro-pause). Do NOT shout to the point of "
        "distortion. Single take, no music or sound effects.",
    ]
    return " ".join(p.strip() for p in parts if p)


def _instruction_hash(instr: str) -> str:
    return "sha1:" + hashlib.sha1(instr.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# WAV decoding helper
# ---------------------------------------------------------------------------


def _decode_wav_bytes(buf: bytes):
    """Return (numpy float32 array, sample_rate) from raw .wav bytes."""
    import soundfile as sf
    import numpy as np
    wav, sr = sf.read(io.BytesIO(buf), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32), int(sr)


def _save_wav(path: str, audio, sr: int) -> None:
    import soundfile as sf
    sf.write(path, audio, sr, subtype="PCM_16")


# ---------------------------------------------------------------------------
# Per-sample TTS
# ---------------------------------------------------------------------------


def _generate_one(record: Dict, audio_dir: str) -> Optional[Dict]:
    label = record["label"]
    sample_id = record["id"]
    text = record["text"]
    voice = _next_voice(label)
    instruction = build_instruction(record)
    instr_hash = _instruction_hash(instruction)

    out_path = os.path.join(audio_dir, label, f"{sample_id}.wav")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        # Already generated - return manifest row directly
        try:
            wav, sr = _decode_wav_bytes(open(out_path, "rb").read())
            duration = len(wav) / sr
        except Exception:
            duration = 0.0
            sr = 16000
        return _manifest_row(record, voice, instr_hash, duration, sr,
                             out_path, model=cfg.TTS_MODEL)

    pool = get_pool("tts")

    def _call(client):
        # The OpenAI Python SDK 1.x exposes audio.speech.create returning
        # a streaming binary response. We read it to memory; clips are
        # short (<10 s).
        return client.audio.speech.create(
            model=cfg.TTS_MODEL,
            voice=voice,
            input=text,
            instructions=instruction,
            response_format="wav",
        )

    try:
        resp = with_pool_backoff(pool, _call)
    except Exception as e:
        print(f"\n  [WARN] TTS failed for {sample_id} ({label}): "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return None

    # Newer SDKs return an object whose .read() yields bytes; older ones
    # support .stream_to_file. Cover both.
    audio_bytes: Optional[bytes] = None
    try:
        # Streaming response object
        audio_bytes = resp.read()  # type: ignore[attr-defined]
    except Exception:
        try:
            buf = io.BytesIO()
            resp.stream_to_file(buf)  # type: ignore[attr-defined]
            audio_bytes = buf.getvalue()
        except Exception:
            audio_bytes = getattr(resp, "content", None)

    if not audio_bytes:
        print(f"\n  [WARN] empty audio for {sample_id}", file=sys.stderr)
        return None

    try:
        wav, sr = _decode_wav_bytes(audio_bytes)
    except Exception as e:
        print(f"\n  [WARN] could not decode wav for {sample_id}: {e}",
              file=sys.stderr)
        return None

    # Resample to 16 kHz if needed (OpenAI returns 24 kHz currently)
    target_sr = 16000
    if sr != target_sr:
        try:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except ImportError:
            pass  # keep original sr; downstream loaders auto-resample

    _save_wav(out_path, wav, sr)
    duration = len(wav) / sr
    return _manifest_row(record, voice, instr_hash, duration, sr,
                         out_path, model=cfg.TTS_MODEL)


def _manifest_row(record: Dict, voice: str, instr_hash: str,
                   duration: float, sr: int, audio_path: str,
                   model: str) -> Dict:
    return {
        "audio_id":            record["id"],
        "label":               record["label"],
        "label_id":            record["label_id"],
        "text":                record["text"],
        "intensity":           record["intensity"],
        "cause":               record["cause"],
        "style":               record["style"],
        "persona":             record["persona"],
        "turn_position":       record["turn_position"],
        "voice":               voice,
        "instruction_hash":    instr_hash,
        "duration_sec":        round(float(duration), 3),
        "sample_rate":         int(sr),
        "judge_score":         record.get("judge_score", ""),
        "judge_intensity_obs": record.get("judge_intensity_obs", ""),
        "judge_natural":       record.get("judge_natural", ""),
        "model":               model,
        "ts":                  time.time(),
        "audio_path":          audio_path.replace("\\", "/"),
    }


# ---------------------------------------------------------------------------
# Manifest helpers (resume-safe)
# ---------------------------------------------------------------------------

_MANIFEST_FIELDS = [
    "audio_id", "label", "label_id", "text",
    "intensity", "cause", "style", "persona", "turn_position",
    "voice", "instruction_hash", "duration_sec", "sample_rate",
    "judge_score", "judge_intensity_obs", "judge_natural",
    "model", "ts", "audio_path",
]


def _load_manifest_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    seen = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "audio_id" in row:
                seen.add(row["audio_id"])
    return seen


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synthesise WAV audio for filtered text utterances."
    )
    p.add_argument("--input",   type=str, default=cfg.TEXT_FILTERED)
    p.add_argument("--manifest", type=str, default=cfg.MANIFEST_OUT)
    p.add_argument("--audio-dir", type=str, default=cfg.AUDIO_DIR)
    p.add_argument("--workers", type=int, default=cfg.AUDIO_CONCURRENCY)
    p.add_argument("--limit",   type=int, default=None)
    p.add_argument("--label",   type=str, default=None,
                   choices=cfg.TARGET_LABELS)
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args(argv)


def _read_jsonl(path: str) -> List[Dict]:
    out: List[Dict] = []
    if not os.path.exists(path):
        return out
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


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    load_dotenv()
    cfg.ensure_dirs()

    print("=" * 72)
    print("  SmartHandover - Synthetic Audio Generator")
    print("=" * 72)
    print(f"  input    : {args.input}")
    print(f"  manifest : {args.manifest}")
    print(f"  audio    : {args.audio_dir}")
    print(f"  workers  : {args.workers}")
    print(f"  model    : {cfg.TTS_MODEL}")
    print(f"  voices   : {len(cfg.TTS_VOICES)}")
    print(f"  resume   : {not args.no_resume}")

    records = _read_jsonl(args.input)
    print(f"  records  : {len(records)}")
    if args.label:
        records = [r for r in records if r["label"] == args.label]
        print(f"  filtered : {len(records)} (label={args.label})")
    if args.limit is not None:
        records = records[:args.limit]
        print(f"  limited  : {len(records)}")
    if not records:
        print("  Nothing to do.")
        return

    if args.no_resume:
        # Truncate manifest
        if os.path.exists(args.manifest):
            os.remove(args.manifest)
        already = set()
    else:
        already = _load_manifest_ids(args.manifest)
    if already:
        print(f"  resuming : {len(already)} already in manifest")
        records = [r for r in records if r["id"] not in already]
        print(f"  remaining: {len(records)}")

    if not records:
        print("\n  Nothing to do.")
        return

    try:
        tts_pool = get_pool("tts")
    except RuntimeError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(2)
    print()
    print(tts_pool.summary())
    print()

    t0 = time.time()
    n_ok, n_fail = 0, 0
    durations: List[float] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_generate_one, r, args.audio_dir): r
                   for r in records}
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
    avg_dur = sum(durations) / len(durations) if durations else 0.0

    print()
    print(f"  ok={n_ok}  fail={n_fail}  elapsed={elapsed:.0f}s  "
          f"({n_ok / max(elapsed, 1):.2f} clips/s)")
    print(f"  avg clip duration: {avg_dur:.2f}s")
    print(f"  manifest -> {args.manifest}")
    print()
    print(tts_pool.summary())

    if n_fail:
        print("\n  Some clips failed - re-run to retry (resume kicks in).")


if __name__ == "__main__":
    main()
