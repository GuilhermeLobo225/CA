"""SmartHandover - Synthetic data pipeline configuration.

Single source of truth for the synthetic dataset. Everything that affects
reproducibility (model, distribution, seed, paths, voices) lives here so
that the generated manifest can record the exact configuration.

Edit the constants below to retarget the pipeline (e.g. drop to 10k).
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Schema (must match src/data/load_meld.py and load_cremad.py)
# ---------------------------------------------------------------------------

TARGET_LABELS = ["anger", "frustration", "sadness", "neutral", "satisfaction"]
TARGET_LABEL2ID = {label: i for i, label in enumerate(TARGET_LABELS)}

# ---------------------------------------------------------------------------
# Target distribution
# ---------------------------------------------------------------------------
#
# Strategy: generate exactly enough synthetic samples per class so that
# MELD_train + synthetic == BALANCE_TARGET_PER_CLASS for every class.
# This produces a perfectly balanced training set after the join.
#
# MELD_train class counts (frozen - see src/data/load_meld.py output):
MELD_TRAIN_COUNTS = {
    "anger":        1_380,
    "frustration":    268,
    "sadness":        683,
    "neutral":      4_709,
    "satisfaction": 1_743,
}

# Target count per class in the COMBINED training set (MELD + synthetic).
# 6000 keeps the synthetic budget close to 20k (the value originally
# scoped) while pushing frustration from 3.1% to 20% of the combined set.
# Override via env var SYNTH_BALANCE_TARGET if you want a smaller / larger run.
BALANCE_TARGET_PER_CLASS = int(
    os.environ.get("SYNTH_BALANCE_TARGET", "6000")
)


def _compute_distribution() -> dict:
    """Return the synthetic-per-class counts implied by BALANCE_TARGET."""
    return {
        label: max(0, BALANCE_TARGET_PER_CLASS - MELD_TRAIN_COUNTS[label])
        for label in MELD_TRAIN_COUNTS
    }


# Allow a manual override if you want to bypass the auto-balance logic
# (set SYNTH_DISTRIBUTION_OVERRIDE='anger=4000,frustration=5500,...').
def _parse_override(env_value: str) -> dict:
    parts = [p.strip() for p in env_value.split(",") if p.strip()]
    out = {}
    for p in parts:
        k, _, v = p.partition("=")
        out[k.strip()] = int(v.strip())
    # Fill missing classes with 0 to keep schema consistent
    for label in MELD_TRAIN_COUNTS:
        out.setdefault(label, 0)
    return out


_override = os.environ.get("SYNTH_DISTRIBUTION_OVERRIDE", "").strip()
DISTRIBUTION = (
    _parse_override(_override) if _override else _compute_distribution()
)
TOTAL_SAMPLES = sum(DISTRIBUTION.values())

# Optional global multiplier for dry runs (e.g. SCALE=0.05 -> 5% of full).
SCALE = float(os.environ.get("SYNTH_SCALE", "1.0"))

# ---------------------------------------------------------------------------
# OpenAI configuration
# ---------------------------------------------------------------------------

# These IDs are stable as of writing. Override at runtime via env vars if
# OpenAI deprecates them.
TEXT_MODEL  = os.environ.get("SYNTH_TEXT_MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.environ.get("SYNTH_JUDGE_MODEL", "mistral-small3.1:latest")
TTS_MODEL   = os.environ.get("SYNTH_TTS_MODEL", "gpt-4o-mini-tts")

# Judge provider:
#   * "ollama" (default) - run a local LLM via Ollama's OpenAI-compatible
#                          endpoint. No bias from generator-as-judge.
#   * "iaedu"            - reuse the IAEDU text pool (gpt-4o judges its own
#                          output - cheap but biased).
#   * "openai"           - use the standard OpenAI API with OPENAI_TTS_API_KEY
#                          (or OPENAI_API_KEY) - paid, fastest.
JUDGE_PROVIDER = os.environ.get("JUDGE_PROVIDER", "ollama").lower()

# Ollama endpoint (only used when JUDGE_PROVIDER=ollama).
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# All available gpt-4o-mini-tts voices. Round-robin across all of them so the
# audio classifier cannot lock onto a single speaker identity.
TTS_VOICES = [
    "alloy", "ash", "ballad", "coral", "echo", "fable",
    "onyx", "nova", "sage", "shimmer", "verse",
]

# ---------------------------------------------------------------------------
# Retry / rate-limit defaults (used by _openai_client)
# ---------------------------------------------------------------------------

MAX_RETRIES         = 6
INITIAL_BACKOFF_SEC = 2.0
MAX_BACKOFF_SEC     = 60.0
REQUEST_TIMEOUT_SEC = 60.0

# Concurrency. OpenAI Tier-1 lets us run ~8-12 parallel chat requests
# without hitting RPM limits. TTS is heavier and slower so default to 4.
TEXT_CONCURRENCY  = int(os.environ.get("SYNTH_TEXT_CONCURRENCY", "8"))
# Ollama runs on a single GPU; running 8 concurrent inferences just queues
# them. 2 keeps the queue short without leaving the GPU idle.
JUDGE_CONCURRENCY = int(
    os.environ.get(
        "SYNTH_JUDGE_CONCURRENCY",
        "2" if JUDGE_PROVIDER == "ollama" else "8",
    )
)
AUDIO_CONCURRENCY = int(os.environ.get("SYNTH_AUDIO_CONCURRENCY", "4"))

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = int(os.environ.get("SYNTH_SEED", "42"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUT_DIR        = os.path.join("data", "synthetic")
TEXT_OUT       = os.path.join(OUT_DIR, "text.jsonl")
TEXT_FILTERED  = os.path.join(OUT_DIR, "text_filtered.jsonl")
TEXT_REJECTED  = os.path.join(OUT_DIR, "text_rejected.jsonl")
AUDIO_DIR      = os.path.join(OUT_DIR, "audio")
MANIFEST_OUT   = os.path.join(OUT_DIR, "manifest.csv")
RUN_LOG        = os.path.join(OUT_DIR, "run_log.json")

# Where the audited validation goes
VALIDATION_OUT = os.path.join(OUT_DIR, "validation_results.csv")
VALIDATION_REPORT = os.path.join(OUT_DIR, "validation_report.json")

# ---------------------------------------------------------------------------
# Quality thresholds (filter stage)
# ---------------------------------------------------------------------------

MIN_WORDS = 4
MAX_WORDS = 60
MIN_JUDGE_SCORE = 3      # 1..5 scale; below this -> rejected
MIN_JUDGE_INTENSITY_AGREE = 1   # |judge_intensity - target_intensity| <= this


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def scaled_distribution() -> dict:
    """Return DISTRIBUTION scaled by SCALE (rounded to int)."""
    return {k: int(round(v * SCALE)) for k, v in DISTRIBUTION.items()}


def ensure_dirs() -> None:
    """Create the output directory tree (idempotent)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    for label in TARGET_LABELS:
        os.makedirs(os.path.join(AUDIO_DIR, label), exist_ok=True)
