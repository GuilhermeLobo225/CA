"""
SmartHandover — Conversation-Level Handover Simulation (Day 10)

Sequentially replays MELD test conversations through the sliding-window
``HandoverDecision`` logic to measure conversation-level effectiveness.

Conversations are reconstructed from the audio file paths exposed by the
``ajyy/MELD_audio`` HuggingFace dataset — each path has the form
``dia<ID>_utt<N>.flac`` so utterances within a dialogue can be sorted by
``N`` and grouped by ``ID``.

Per-utterance meta-classifier probabilities are **reused** from the Day 6
artefacts (``data/processed/ensemble_features_test.csv`` +
``checkpoints/meta_classifier.pkl``) — no Whisper / RoBERTa / SpeechBrain
inference happens here, so the simulation is fast (~seconds).

Metrics reported
----------------
* Conversation-level handover recall — fraction of conversations that
  contained at least one frustration / anger utterance and that the system
  flagged at any point.
* Conversation-level false-handover rate — fraction of "clean" conversations
  (no frustration / anger ground-truth utterance) that nonetheless triggered
  a handover.
* A per-conversation CSV with the first trigger index and reason.

Run:

    python -m src.decision.simulate_handover
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict, Counter
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

# Project imports -----------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.classifiers.ensemble_trainer import (  # noqa: E402
    CKPT_DIR,
    DATA_DIR,
    FEATURE_COLUMNS,
    TARGET_LABELS,
)
from src.decision.handover import HandoverDecision  # noqa: E402

CONFIG_DIR = "configs"
THRESHOLD_JSON = os.path.join(CONFIG_DIR, "handover_threshold.json")
DEFAULT_THRESHOLD = 0.6
DEFAULT_WINDOW = 3

# Emotions that justify a handover (ground-truth labels)
HANDOVER_EMOTIONS = {"anger", "frustration"}

# Regex to pull dialogue / utterance indices from a MELD path
_DIA_UTT_RE = re.compile(r"dia(\d+)_utt(\d+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_threshold() -> float:
    """Pick up the Day-8 optimal threshold if present, else use fallback."""
    if os.path.exists(THRESHOLD_JSON):
        try:
            with open(THRESHOLD_JSON, "r", encoding="utf-8") as f:
                return float(json.load(f)["optimal_threshold"])
        except (KeyError, ValueError, OSError):
            pass
    print(f"  [WARN] No threshold JSON at {THRESHOLD_JSON}, "
          f"using default={DEFAULT_THRESHOLD}")
    return DEFAULT_THRESHOLD


def _load_test_features() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "ensemble_features_test.csv")
    return pd.read_csv(path)


def _load_meta_classifier():
    bundle = joblib.load(os.path.join(CKPT_DIR, "meta_classifier.pkl"))
    return bundle["model"], bundle.get("name", type(bundle["model"]).__name__)


def _parse_dialogue_utterance(path: str) -> Tuple[int, int]:
    m = _DIA_UTT_RE.search(path)
    if not m:
        raise ValueError(f"Cannot parse dia/utt indices from path: {path}")
    return int(m.group(1)), int(m.group(2))


def _load_meld_conversation_map() -> List[Tuple[int, int, str]]:
    """Return (dialogue_id, utterance_id, true_emotion) for MELD test rows.

    Row order matches the one used when the prediction CSVs were built,
    so the result can be zipped with ``ensemble_features_test.csv``.
    """
    from src.data.load_meld import load_meld  # local import to keep startup fast
    ds = load_meld(split="test", streaming=False)

    out = []
    for example in ds:
        dia, utt = _parse_dialogue_utterance(example["path"])
        out.append((dia, utt, example["target_emotion"]))
    return out


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def build_conversations(
    features_df: pd.DataFrame, meta_meld: List[Tuple[int, int, str]],
    probs: np.ndarray,
) -> Dict[int, List[Dict[str, Any]]]:
    """Group per-utterance rows into conversations keyed by dialogue_id.

    Each conversation is a list of utterance dicts sorted by utterance_id.
    """
    if len(meta_meld) != len(features_df):
        print(f"  [WARN] length mismatch: MELD={len(meta_meld)} vs "
              f"features={len(features_df)} — truncating to the shorter.")
    n = min(len(meta_meld), len(features_df))

    convs: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for i in range(n):
        dia, utt, true_emo = meta_meld[i]
        row = features_df.iloc[i]
        convs[dia].append({
            "dialogue_id": dia,
            "utterance_id": utt,
            "true_emotion": true_emo,
            "audio_id":     row.get("audio_id", f"test_{i:05d}"),
            "text":         row.get("text", ""),
            "meta_probs":   {
                TARGET_LABELS[k]: float(probs[i, k])
                for k in range(len(TARGET_LABELS))
            },
        })

    # Ensure each conversation is ordered by utterance_id
    for dia in convs:
        convs[dia].sort(key=lambda u: u["utterance_id"])
    return convs


def simulate(
    conversations: Dict[int, List[Dict[str, Any]]],
    threshold: float,
    window_size: int,
) -> pd.DataFrame:
    """Run every conversation through ``HandoverDecision``.

    Returns a per-conversation DataFrame.
    """
    decider = HandoverDecision(threshold=threshold, window_size=window_size)

    rows = []
    for dia_id, utters in conversations.items():
        true_emotions = [u["true_emotion"] for u in utters]
        has_negative = any(e in HANDOVER_EMOTIONS for e in true_emotions)
        first_neg_idx = next(
            (i for i, e in enumerate(true_emotions)
             if e in HANDOVER_EMOTIONS),
            None,
        )
        n_negative = sum(1 for e in true_emotions if e in HANDOVER_EMOTIONS)

        result = decider.process_stream(utters)
        if result is not None:
            trigger_idx, reason = result
        else:
            trigger_idx, reason = -1, HandoverDecision.REASON_OK

        # Did the system correctly trigger? (only meaningful for positives)
        caught = has_negative and (trigger_idx >= 0)
        false_handover = (not has_negative) and (trigger_idx >= 0)

        # Timeliness — how many utterances *after* the first real negative
        # did the system take to flag? Negative means it pre-emptively
        # flagged before any negative utterance (often a false positive).
        if has_negative and trigger_idx >= 0:
            latency_utt = trigger_idx - (first_neg_idx or 0)
        else:
            latency_utt = None

        rows.append({
            "dialogue_id":     dia_id,
            "length":          len(utters),
            "has_negative":    has_negative,
            "n_negative_utts": n_negative,
            "first_neg_utt":   first_neg_idx if first_neg_idx is not None else -1,
            "trigger_utt":    trigger_idx,
            "trigger_reason": reason,
            "caught":         bool(caught),
            "false_handover": bool(false_handover),
            "latency_utt":    latency_utt,
        })

    return pd.DataFrame(rows)


def summarise(sim_df: pd.DataFrame) -> Dict[str, Any]:
    """Aggregate conversation-level metrics from the per-conversation DF."""
    n_total        = len(sim_df)
    n_positive     = int(sim_df["has_negative"].sum())
    n_negative     = n_total - n_positive
    n_caught       = int(sim_df["caught"].sum())
    n_false        = int(sim_df["false_handover"].sum())
    n_missed       = n_positive - n_caught
    n_triggered    = int((sim_df["trigger_utt"] >= 0).sum())

    recall = n_caught / n_positive if n_positive else 0.0
    false_rate = n_false / n_negative if n_negative else 0.0
    precision = n_caught / n_triggered if n_triggered else 0.0

    # Only averaged over conversations where we caught it
    caught_df = sim_df[sim_df["caught"] & sim_df["latency_utt"].notna()]
    mean_latency = float(caught_df["latency_utt"].mean()) if len(caught_df) else float("nan")

    reason_counts: Counter = Counter(
        sim_df[sim_df["trigger_utt"] >= 0]["trigger_reason"].tolist()
    )

    return {
        "n_conversations":      n_total,
        "n_with_negative":      n_positive,
        "n_clean":              n_negative,
        "n_caught":             n_caught,
        "n_missed":             n_missed,
        "n_false_handovers":    n_false,
        "n_triggered_any":      n_triggered,
        "conv_recall":          recall,
        "conv_precision":       precision,
        "conv_false_rate":      false_rate,
        "mean_catch_latency_utt": mean_latency,
        "trigger_reason_counts": dict(reason_counts),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("  SmartHandover — Day 10: Conversation-Level Simulation")
    print("=" * 60)

    threshold = _load_threshold()
    window_size = DEFAULT_WINDOW
    print(f"  Threshold   : {threshold:.3f}  (from {THRESHOLD_JSON})")
    print(f"  Window size : {window_size}")

    # --- Load predictions ------------------------------------------------
    features_df = _load_test_features()
    meta_model, meta_name = _load_meta_classifier()
    print(f"  Meta model  : {meta_name}  ({len(features_df)} test utterances)")

    X = features_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    if hasattr(meta_model, "predict_proba"):
        probs = meta_model.predict_proba(X)
    else:
        preds = meta_model.predict(X)
        probs = np.eye(len(TARGET_LABELS))[preds]

    # --- Build conversations --------------------------------------------
    print("\n  Grouping MELD test utterances by dialogue_id ...")
    meld_meta = _load_meld_conversation_map()
    conversations = build_conversations(features_df, meld_meta, probs)
    print(f"  {len(conversations)} conversations, "
          f"avg length={np.mean([len(v) for v in conversations.values()]):.1f} utts")

    # --- Run simulation --------------------------------------------------
    print("\n  Simulating handover decisions ...")
    sim_df = simulate(conversations, threshold=threshold,
                      window_size=window_size)

    out_csv = os.path.join(DATA_DIR, "handover_simulation.csv")
    sim_df.to_csv(out_csv, index=False)
    print(f"  Per-conversation results saved -> {out_csv}")

    # --- Summary ---------------------------------------------------------
    summary = summarise(sim_df)

    print("\n" + "=" * 60)
    print("  Conversation-Level Handover Metrics")
    print("=" * 60)
    print(f"  Total conversations             : {summary['n_conversations']}")
    print(f"  w/ frustration or anger         : {summary['n_with_negative']}")
    print(f"  w/o frustration or anger        : {summary['n_clean']}")
    print()
    print(f"  Conversations caught            : {summary['n_caught']} / "
          f"{summary['n_with_negative']}  "
          f"(recall = {summary['conv_recall']*100:.1f}%)")
    print(f"  Conversations missed            : {summary['n_missed']}")
    print(f"  False handovers (on clean convs): {summary['n_false_handovers']} / "
          f"{summary['n_clean']}  "
          f"(rate = {summary['conv_false_rate']*100:.1f}%)")
    print(f"  Handovers triggered (total)     : {summary['n_triggered_any']}")
    print(f"  Conv-level handover precision   : "
          f"{summary['conv_precision']*100:.1f}%")
    print(f"  Mean catch latency (utterances) : "
          f"{summary['mean_catch_latency_utt']:.2f}")
    print()
    print("  Trigger reason breakdown:")
    for reason, cnt in sorted(
        summary["trigger_reason_counts"].items(),
        key=lambda kv: -kv[1],
    ):
        print(f"    {reason:<30s} {cnt}")

    # Persist summary JSON
    out_json = os.path.join(DATA_DIR, "handover_simulation_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "threshold": threshold,
            "window_size": window_size,
            **summary,
        }, f, indent=2)
    print(f"\n  Summary saved -> {out_json}")

    print("\nDone.")


if __name__ == "__main__":
    main()
