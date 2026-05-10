#!/usr/bin/env python3
"""SmartHandover - Day F17: Conversation-level handover simulation (v4).

Replays MELD test conversations through the v4 meta-classifier +
weighted-anger-frustration threshold rule:

    handover_score(t) = w_anger * P(anger; t) + w_frust * P(frust; t)
    fire if handover_score > threshold  (instant rule)
    OR if mean of last W scores > threshold * 0.7   (sustained rule)

Default values come from ``configs/handover_threshold_v4.json``:
    w_anger=0.6, w_frust=0.4, threshold=0.20

Outputs
-------
* ``data/processed/handover_simulation_v4.csv`` (one row per conversation)
* ``data/processed/handover_simulation_v4_summary.json``

This is the cabeçalho metric for the report:
"At conversation level, the system catches X% of dialogues that contain
frustrated/angry utterances, with Y% false-handover rate."
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classifiers.ensemble_trainer import (  # noqa: E402
    DATA_DIR, CKPT_DIR, TARGET_LABELS, TARGET_LABEL2ID,
)


SUFFIX = "_v4"
CONFIG_DIR = "configs"
THRESHOLD_JSON = os.path.join(CONFIG_DIR, f"handover_threshold{SUFFIX}.json")
META_CKPT = os.path.join(CKPT_DIR, f"meta_classifier{SUFFIX}_calibrated.pkl")
FEATURES_CSV = os.path.join(DATA_DIR, f"ensemble_features_test{SUFFIX}.csv")

OUT_CSV = os.path.join(DATA_DIR, f"handover_simulation{SUFFIX}.csv")
OUT_JSON = os.path.join(DATA_DIR, f"handover_simulation{SUFFIX}_summary.json")

DEFAULT_WINDOW = 3
HANDOVER_EMOTIONS = {"anger", "frustration"}

_DIA_UTT_RE = re.compile(r"dia(\d+)_utt(\d+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_config() -> Tuple[float, float, float]:
    if not os.path.exists(THRESHOLD_JSON):
        print(f"[ERROR] {THRESHOLD_JSON} not found. Run Day F16 first.")
        sys.exit(1)
    with open(THRESHOLD_JSON, "r", encoding="utf-8") as f:
        c = json.load(f)
    return (float(c["weights"]["w_anger"]),
            float(c["weights"]["w_frust"]),
            float(c["optimal_threshold"]))


def _load_meld_conversation_map() -> List[Tuple[int, int, str]]:
    """Returns (dialogue_id, utterance_id, true_emotion) per MELD test row."""
    from src.data.load_meld import load_meld
    ds = load_meld(split="test", streaming=False)
    out = []
    for example in ds:
        m = _DIA_UTT_RE.search(example["path"])
        if not m:
            raise ValueError(f"Cannot parse path: {example['path']}")
        out.append((int(m.group(1)), int(m.group(2)),
                     example["target_emotion"]))
    return out


# ---------------------------------------------------------------------------
# Simulation logic (v4 weighted rule)
# ---------------------------------------------------------------------------


def conversation_decision(
    handover_scores: List[float],
    threshold: float,
    window_size: int = DEFAULT_WINDOW,
) -> Tuple[int, str]:
    """Apply two rules sequentially over ``handover_scores`` (one per
    utterance, in order). Returns (trigger_index, reason). If no rule
    fires, returns (-1, 'ok').
    """
    history: List[float] = []
    for idx, s in enumerate(handover_scores):
        history.append(s)
        # Rule 1: instant strong score
        if s > threshold:
            return idx, "instant_strong_emotion"
        # Rule 2: sustained negative trend (full window)
        if len(history) >= window_size:
            window = history[-window_size:]
            if (sum(window) / window_size) > threshold * 0.7:
                return idx, "sustained_negative_trend"
    return -1, "ok"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("  Day F17 - Conversation-level handover simulation (v4)")
    print("=" * 70)

    w_anger, w_frust, threshold = _load_config()
    print(f"  weights      : anger={w_anger:.2f}, frust={w_frust:.2f}")
    print(f"  threshold    : {threshold:.3f}")
    print(f"  window size  : {DEFAULT_WINDOW}")
    print(f"  meta model   : {META_CKPT}")
    print(f"  features     : {FEATURES_CSV}")

    # --- Load model + features ------------------------------------------
    bundle = joblib.load(META_CKPT)
    model = bundle["model"]
    feat_cols = bundle.get("feature_columns")
    if not feat_cols:
        # Fallback to v4 schema
        from scripts.run_dayF16_ensemble_v4 import FEATURE_COLUMNS_V4
        feat_cols = FEATURE_COLUMNS_V4
    print(f"  v4 features  : {len(feat_cols)} dims")

    df = pd.read_csv(FEATURES_CSV)
    X = df[feat_cols].to_numpy(dtype=np.float32)
    probs = model.predict_proba(X)

    a_id = TARGET_LABEL2ID["anger"]
    f_id = TARGET_LABEL2ID["frustration"]
    handover_scores = w_anger * probs[:, a_id] + w_frust * probs[:, f_id]

    # --- Build conversations --------------------------------------------
    print("\n  Loading MELD test conversation map ...")
    meld_meta = _load_meld_conversation_map()
    n = min(len(meld_meta), len(df))
    if len(meld_meta) != len(df):
        print(f"  [WARN] mismatch: meld={len(meld_meta)} vs feat={len(df)}; "
              f"truncating to {n}")

    convs: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for i in range(n):
        dia, utt, true_emo = meld_meta[i]
        convs[dia].append({
            "utterance_id": utt,
            "true_emotion": true_emo,
            "score":        float(handover_scores[i]),
        })
    for dia in convs:
        convs[dia].sort(key=lambda u: u["utterance_id"])

    print(f"  {len(convs)} conversations, "
          f"avg utterances={np.mean([len(v) for v in convs.values()]):.1f}")

    # --- Simulate --------------------------------------------------------
    print("\n  Simulating handover decisions per conversation ...")
    rows = []
    for dia_id, utters in convs.items():
        true_emos = [u["true_emotion"] for u in utters]
        scores = [u["score"] for u in utters]
        has_neg = any(e in HANDOVER_EMOTIONS for e in true_emos)
        first_neg_idx = next(
            (i for i, e in enumerate(true_emos) if e in HANDOVER_EMOTIONS),
            None,
        )
        n_neg = sum(1 for e in true_emos if e in HANDOVER_EMOTIONS)

        trigger_idx, reason = conversation_decision(scores, threshold)
        caught         = has_neg and trigger_idx >= 0
        false_handover = (not has_neg) and trigger_idx >= 0
        latency = (trigger_idx - first_neg_idx
                   if (has_neg and trigger_idx >= 0
                       and first_neg_idx is not None)
                   else None)

        rows.append({
            "dialogue_id":     dia_id,
            "length":          len(utters),
            "has_negative":    has_neg,
            "n_negative_utts": n_neg,
            "first_neg_utt":   first_neg_idx if first_neg_idx is not None else -1,
            "trigger_utt":     trigger_idx,
            "trigger_reason":  reason,
            "trigger_score":   scores[trigger_idx] if trigger_idx >= 0 else 0.0,
            "max_score":       max(scores) if scores else 0.0,
            "caught":          bool(caught),
            "false_handover":  bool(false_handover),
            "latency_utt":     latency,
        })

    sim_df = pd.DataFrame(rows)
    sim_df.to_csv(OUT_CSV, index=False)
    print(f"  Per-conversation -> {OUT_CSV}")

    # --- Summary ---------------------------------------------------------
    n_total     = len(sim_df)
    n_pos       = int(sim_df["has_negative"].sum())
    n_neg       = n_total - n_pos
    n_caught    = int(sim_df["caught"].sum())
    n_false     = int(sim_df["false_handover"].sum())
    n_missed    = n_pos - n_caught
    n_triggered = int((sim_df["trigger_utt"] >= 0).sum())

    recall      = n_caught / n_pos if n_pos else 0.0
    false_rate  = n_false / n_neg if n_neg else 0.0
    precision   = n_caught / n_triggered if n_triggered else 0.0

    caught_df = sim_df[sim_df["caught"] & sim_df["latency_utt"].notna()]
    mean_latency = (float(caught_df["latency_utt"].mean())
                     if len(caught_df) else float("nan"))

    reason_counts = Counter(
        sim_df[sim_df["trigger_utt"] >= 0]["trigger_reason"].tolist()
    )

    print()
    print("=" * 70)
    print("  v4 Conversation-level metrics")
    print("=" * 70)
    print(f"  Total conversations              : {n_total}")
    print(f"  With anger/frustration           : {n_pos}")
    print(f"  Clean conversations              : {n_neg}")
    print()
    print(f"  Caught                           : {n_caught} / {n_pos}   "
          f"(recall = {recall*100:.1f}%)")
    print(f"  Missed                           : {n_missed}")
    print(f"  False handovers on clean convs   : {n_false} / {n_neg}   "
          f"(rate = {false_rate*100:.1f}%)")
    print(f"  Total handovers triggered        : {n_triggered}")
    print(f"  Conv-level precision             : {precision*100:.1f}%")
    print(f"  Mean catch latency (utterances)  : {mean_latency:.2f}")
    print()
    print("  Trigger reason breakdown:")
    for r, c in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {r:<30s} {c}")

    # --- Compare v1 vs v4 (if v1 summary exists) -------------------------
    v1_path = os.path.join(DATA_DIR, "handover_simulation_summary.json")
    if os.path.exists(v1_path):
        with open(v1_path, "r", encoding="utf-8") as f:
            v1 = json.load(f)
        print()
        print("=" * 70)
        print("  v1 vs v4 (conversation-level)")
        print("=" * 70)
        print(f"  {'metric':<32s} {'v1':>10s} {'v4':>10s} {'delta':>10s}")
        print("  " + "-" * 65)
        for label, v1key, v4val in [
            ("recall",            "conv_recall",       recall),
            ("precision",         "conv_precision",    precision),
            ("false handover rate","conv_false_rate",   false_rate),
            ("# conv. caught",    "n_caught",          n_caught),
            ("# false handovers", "n_false_handovers", n_false),
        ]:
            v1val = v1.get(v1key, float("nan"))
            try:
                delta = v4val - v1val
                fmt = "{:>10.3f}" if isinstance(v4val, float) else "{:>10d}"
                print(f"  {label:<32s} {fmt.format(v1val)} "
                      f"{fmt.format(v4val)} {delta:>+10.3f}")
            except (TypeError, ValueError):
                pass

    summary = {
        "weights":       {"w_anger": w_anger, "w_frust": w_frust},
        "threshold":     threshold,
        "window_size":   DEFAULT_WINDOW,
        "n_conversations": n_total,
        "n_with_negative": n_pos,
        "n_clean":         n_neg,
        "n_caught":        n_caught,
        "n_missed":        n_missed,
        "n_false_handovers": n_false,
        "n_triggered_any": n_triggered,
        "conv_recall":     recall,
        "conv_precision":  precision,
        "conv_false_rate": false_rate,
        "mean_catch_latency_utt": mean_latency,
        "trigger_reason_counts":  dict(reason_counts),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary -> {OUT_JSON}")


if __name__ == "__main__":
    main()
