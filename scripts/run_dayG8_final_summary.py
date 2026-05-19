#!/usr/bin/env python3
"""SmartHandover - Day G8: Final consolidation.

Reads every v5 summary produced by Phases 1-5 and the v4 baselines, and
emits ``data/processed/v5_final_summary.json`` with the single source of
truth for the README. Also writes a plain-text report block to stdout.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join("data", "processed")


def _read(path: str, default=None) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    print("=" * 78)
    print("  Day G8 - Final consolidation")
    print("=" * 78)

    v4_meta   = _read(os.path.join(DATA_DIR, "meta_classifier_v4_summary.json"), {})
    v4_handov = _read(os.path.join(DATA_DIR, "handover_simulation_v4_summary.json"), {})
    v5_meta   = _read(os.path.join(DATA_DIR, "meta_classifier_v5_summary.json"), {})
    v5_handov = _read(os.path.join(DATA_DIR, "handover_simulation_v5_summary.json"), {})
    dayG1     = _read(os.path.join(DATA_DIR, "dayG1_summary.json"), {})
    dayG2     = _read(os.path.join(DATA_DIR, "dayG2_summary.json"), {})
    dayG3     = _read(os.path.join(DATA_DIR, "dayG3_summary.json"), {})

    # Per-utterance MELD test block
    v4_cal_test = v4_meta.get("calibrated_test", {})
    v5_cal_test = v5_meta.get("calibrated_test", {})
    winner = v5_meta.get("best_name", "n/a")
    v5_wins = v5_meta.get("comparison_vs_v4", {}).get("v5_beats_v4", False)

    delta_lines = []
    for k in ("frust_recall", "macro_f1", "weighted_f1", "frust_f1"):
        v4_v = v4_cal_test.get(k, float("nan"))
        v5_v = v5_cal_test.get(k, float("nan"))
        try:
            delta_lines.append(f"{k}: v4={v4_v:.4f} -> v5={v5_v:.4f}  ({v5_v - v4_v:+.4f})")
        except (TypeError, ValueError):
            pass
    delta_summary = " | ".join(delta_lines)

    techniques_used = ["smote_k5"]
    techniques_failed = []
    if dayG1.get("aborted"):
        techniques_failed.append("focal_loss (aborted: %s)" % dayG1.get("abort_reason", "n/a"))
    else:
        techniques_used.append("focal_loss_gamma=%.1f" % dayG1.get("config", {}).get("gamma", 2.0)
                                if dayG1 else "weighted_ce")
    if not dayG2.get("v5_used", False):
        techniques_failed.append("v5 RoBERTa predictions did not exceed v4 (Phase 2 fell back to v4 prob_*)")

    if winner.startswith("XGB"):
        techniques_used.append("xgboost (v5 grid winner)")
    elif winner == "Stacking":
        techniques_used.append("stacking (LR + XGB soft-vote)")
    elif winner == "MLP_balanced":
        techniques_used.append("mlp (v5 winner)")
    elif winner == "LR_balanced":
        techniques_used.append("logistic_regression (v5 winner)")
    cm = v5_meta.get("calibration_method")
    if cm:
        techniques_used.append(f"{cm}_calibration")

    if not v5_wins:
        techniques_failed.append(
            "v5 meta-classifier did not beat v4 on 0.5*frust_recall + 0.5*macro_f1"
        )

    out = {
        "per_utterance_meld_test": {
            "v4_calibrated": v4_cal_test,
            "v5_winner":     v5_cal_test,
            "winner_name":   winner,
            "calibration_method": v5_meta.get("calibration_method"),
            "selection_rule":     v5_meta.get("selection_rule"),
            "delta_summary":      delta_summary,
            "v5_beats_v4_composite": bool(v5_wins),
        },
        "handover_utterance_level": {
            "v4": _read(
                os.path.join("configs", "handover_threshold_v4.json"), {}),
            "v5": _read(
                os.path.join("configs", "handover_threshold_v5.json"), {}),
        },
        "handover_conversation_level": {
            "v4": v4_handov,
            "v5": v5_handov,
        },
        "techniques_actually_used_in_v5": techniques_used,
        "techniques_attempted_but_did_not_help": techniques_failed,
        "honesty_notes": [
            "Frustration recall is upper-bounded by the MELD fear->frustration "
            "mapping noise: most MELD 'fear' utterances are sitcom startle "
            "('please don't hurt me'), not customer-support frustration.",
            "MELD is a sitcom dataset (Friends). Numbers here are not predictive "
            "of contact-centre performance.",
            "The conversation-level false-handover rate on clean conversations is "
            "high (see v4/v5 handover_simulation summaries). This is a real "
            "operational concern, not a measurement artefact.",
            ("Phase 1 (focal loss) did not clear its stop bar; v5 features keep "
              "the v4 RoBERTa prob_* columns.") if not dayG2.get("v5_used", False)
              else "Phase 1 (focal loss) produced new RoBERTa probabilities used in v5 features.",
        ],
        "phase_sources": {
            "dayG1_summary":  os.path.join(DATA_DIR, "dayG1_summary.json"),
            "dayG2_summary":  os.path.join(DATA_DIR, "dayG2_summary.json"),
            "dayG3_summary":  os.path.join(DATA_DIR, "dayG3_summary.json"),
            "meta_v5_summary": os.path.join(DATA_DIR, "meta_classifier_v5_summary.json"),
            "handover_v5":    os.path.join(DATA_DIR, "handover_simulation_v5_summary.json"),
        },
    }

    out_path = os.path.join(DATA_DIR, "v5_final_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  v5_final_summary -> {out_path}")

    print("\n=== Final per-utterance MELD test ===")
    for k in ("weighted_f1", "macro_f1", "frust_recall", "frust_f1"):
        v4v = v4_cal_test.get(k, float("nan"))
        v5v = v5_cal_test.get(k, float("nan"))
        try:
            print(f"  {k:<14s} v4={v4v:.4f}  v5={v5v:.4f}  delta={v5v-v4v:+.4f}")
        except Exception:
            pass
    print(f"\n  v5 winner: {winner}")
    print(f"  v5 beats v4 composite: {v5_wins}")
    print("\n=== Final conversation-level ===")
    for k in ("conv_recall", "conv_precision", "conv_false_rate",
                "mean_catch_latency_utt"):
        v4v = v4_handov.get(k, float("nan"))
        v5v = v5_handov.get(k, float("nan"))
        try:
            print(f"  {k:<26s} v4={v4v:.4f}  v5={v5v:.4f}  delta={v5v-v4v:+.4f}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
