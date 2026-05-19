#!/usr/bin/env python3
"""SmartHandover - Day G5: Conversation-level handover simulation (v5).

Mirrors ``scripts/run_dayF17_simulate_handover_v4.py`` but loads the v5
calibrated meta-classifier and the v5 threshold config (Rule B primary).

Outputs
-------
* ``data/processed/handover_simulation_v5.csv``
* ``data/processed/handover_simulation_v5_summary.json``
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
    DATA_DIR,
    CKPT_DIR,
    TARGET_LABEL2ID,
)
from src.classifiers.stacking import StackingSoftVote  # noqa: F401, E402

SUFFIX = "_v5"
THRESHOLD_JSON = os.path.join("configs", f"handover_threshold{SUFFIX}.json")
META_CKPT = os.path.join(CKPT_DIR, f"meta_classifier{SUFFIX}_calibrated.pkl")
FEATURES_CSV = os.path.join(DATA_DIR, f"ensemble_features_test{SUFFIX}.csv")
OUT_CSV = os.path.join(DATA_DIR, f"handover_simulation{SUFFIX}.csv")
OUT_JSON = os.path.join(DATA_DIR, f"handover_simulation{SUFFIX}_summary.json")

DEFAULT_WINDOW = 3
HANDOVER_EMOTIONS = {"anger", "frustration"}
_DIA_UTT_RE = re.compile(r"dia(\d+)_utt(\d+)", re.IGNORECASE)


def _load_config() -> Tuple[float, float, float]:
    if not os.path.exists(THRESHOLD_JSON):
        print(f"[ERROR] {THRESHOLD_JSON} not found. Run dayG4 first.")
        sys.exit(1)
    with open(THRESHOLD_JSON, "r", encoding="utf-8") as f:
        c = json.load(f)
    return (float(c["weights"]["w_anger"]),
            float(c["weights"]["w_frust"]),
            float(c["optimal_threshold"]))


def _load_meld_conv_map() -> List[Tuple[int, int, str]]:
    from src.data.load_meld import load_meld
    ds = load_meld(split="test", streaming=False)
    out = []
    for ex in ds:
        m = _DIA_UTT_RE.search(ex["path"])
        if not m:
            raise ValueError(f"Cannot parse path: {ex['path']}")
        out.append((int(m.group(1)), int(m.group(2)),
                     ex["target_emotion"]))
    return out


def conversation_decision(
    scores: List[float], threshold: float, window: int = DEFAULT_WINDOW,
) -> Tuple[int, str]:
    history: List[float] = []
    for idx, s in enumerate(scores):
        history.append(s)
        if s > threshold:
            return idx, "instant_strong_emotion"
        if len(history) >= window:
            recent = history[-window:]
            if (sum(recent) / window) > threshold * 0.7:
                return idx, "sustained_negative_trend"
    return -1, "ok"


def main() -> None:
    print("=" * 70)
    print("  Day G5 - v5 conversation-level handover simulation")
    print("=" * 70)

    w_anger, w_frust, threshold = _load_config()
    print(f"  weights : anger={w_anger:.2f}, frust={w_frust:.2f}")
    print(f"  threshold: {threshold:.4f}")
    print(f"  meta    : {META_CKPT}")
    print(f"  features: {FEATURES_CSV}")

    bundle = joblib.load(META_CKPT)
    model = bundle["model"]
    feat_cols = bundle.get("feature_columns")
    print(f"  v5 feature dims: {len(feat_cols)}")

    df = pd.read_csv(FEATURES_CSV)
    X = df[feat_cols].to_numpy(dtype=np.float32)
    probs = model.predict_proba(X)
    a_id = TARGET_LABEL2ID["anger"]
    f_id = TARGET_LABEL2ID["frustration"]
    handover_scores = w_anger * probs[:, a_id] + w_frust * probs[:, f_id]

    print("\n  Loading MELD test conversation map ...")
    meld = _load_meld_conv_map()
    n = min(len(meld), len(df))
    if len(meld) != len(df):
        print(f"  [WARN] mismatch: meld={len(meld)} vs feat={len(df)}; "
              f"truncating to {n}")

    convs: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for i in range(n):
        dia, utt, true_emo = meld[i]
        convs[dia].append({"utterance_id": utt,
                              "true_emotion": true_emo,
                              "score": float(handover_scores[i])})
    for dia in convs:
        convs[dia].sort(key=lambda u: u["utterance_id"])

    print(f"  {len(convs)} conversations")

    rows = []
    for dia_id, utters in convs.items():
        true_emos = [u["true_emotion"] for u in utters]
        scores = [u["score"] for u in utters]
        has_neg = any(e in HANDOVER_EMOTIONS for e in true_emos)
        first_neg = next((i for i, e in enumerate(true_emos)
                            if e in HANDOVER_EMOTIONS), None)
        n_neg = sum(1 for e in true_emos if e in HANDOVER_EMOTIONS)

        trigger_idx, reason = conversation_decision(scores, threshold)
        caught = has_neg and trigger_idx >= 0
        false_handover = (not has_neg) and trigger_idx >= 0
        latency = (trigger_idx - first_neg
                    if (has_neg and trigger_idx >= 0 and first_neg is not None)
                    else None)
        rows.append({
            "dialogue_id":     dia_id,
            "length":          len(utters),
            "has_negative":    has_neg,
            "n_negative_utts": n_neg,
            "first_neg_utt":   first_neg if first_neg is not None else -1,
            "trigger_utt":     trigger_idx,
            "trigger_reason":  reason,
            "trigger_score":   scores[trigger_idx] if trigger_idx >= 0 else 0.0,
            "max_score":       max(scores) if scores else 0.0,
            "caught":          bool(caught),
            "false_handover":  bool(false_handover),
            "latency_utt":     latency,
        })
    sim = pd.DataFrame(rows)
    sim.to_csv(OUT_CSV, index=False)
    print(f"  per-conversation -> {OUT_CSV}")

    n_total = len(sim)
    n_pos = int(sim["has_negative"].sum())
    n_neg = n_total - n_pos
    n_caught = int(sim["caught"].sum())
    n_false = int(sim["false_handover"].sum())
    n_missed = n_pos - n_caught
    n_trig = int((sim["trigger_utt"] >= 0).sum())

    recall = n_caught / n_pos if n_pos else 0.0
    precision = n_caught / n_trig if n_trig else 0.0
    false_rate = n_false / n_neg if n_neg else 0.0
    caught_df = sim[sim["caught"] & sim["latency_utt"].notna()]
    mean_latency = (float(caught_df["latency_utt"].mean()) if len(caught_df) else float("nan"))
    reasons = Counter(sim[sim["trigger_utt"] >= 0]["trigger_reason"].tolist())

    print()
    print("=" * 70)
    print("  v5 conversation-level metrics")
    print("=" * 70)
    print(f"  total convs: {n_total}, with negative: {n_pos}, clean: {n_neg}")
    print(f"  caught     : {n_caught}/{n_pos}  (recall={recall*100:.1f}%)")
    print(f"  false      : {n_false}/{n_neg}  (rate={false_rate*100:.1f}%)")
    print(f"  precision  : {precision*100:.1f}%")
    print(f"  latency    : {mean_latency:.2f} utterances")

    summary = {
        "weights":               {"w_anger": w_anger, "w_frust": w_frust},
        "threshold":             threshold,
        "window_size":           DEFAULT_WINDOW,
        "n_conversations":       n_total,
        "n_with_negative":       n_pos,
        "n_clean":               n_neg,
        "n_caught":              n_caught,
        "n_missed":              n_missed,
        "n_false_handovers":     n_false,
        "n_triggered_any":       n_trig,
        "conv_recall":           recall,
        "conv_precision":        precision,
        "conv_false_rate":       false_rate,
        "mean_catch_latency_utt": mean_latency,
        "trigger_reason_counts": dict(reasons),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary -> {OUT_JSON}")

    v4_path = os.path.join(DATA_DIR, "handover_simulation_v4_summary.json")
    if os.path.exists(v4_path):
        with open(v4_path, "r", encoding="utf-8") as f:
            v4 = json.load(f)
        print()
        print("  v4 -> v5 comparison")
        print(f"  {'metric':<26s} {'v4':>10s} {'v5':>10s} {'delta':>10s}")
        for label, k, v5v in [
            ("conv_recall",      "conv_recall",       recall),
            ("conv_precision",   "conv_precision",    precision),
            ("conv_false_rate",  "conv_false_rate",   false_rate),
            ("mean_latency_utt", "mean_catch_latency_utt", mean_latency),
        ]:
            v4v = v4.get(k, float("nan"))
            try:
                d = v5v - v4v
                print(f"  {label:<26s} {v4v:>10.4f} {v5v:>10.4f} {d:>+10.4f}")
            except (TypeError, ValueError):
                pass


if __name__ == "__main__":
    main()
