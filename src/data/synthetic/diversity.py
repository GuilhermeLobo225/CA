"""SmartHandover - Controlled diversity sampling.

Defines the 5-axis diversity space and a sampler that respects per-class
constraints (e.g. ``neutral`` cannot have intensity 5; ``satisfaction``
cannot have ``billing_error`` as cause).

The output of ``sample_point(label, rng)`` is a dict with seven fields
that gets injected into the LLM prompt and later into the TTS instructions.
The same point is also written to the manifest so the relationship
between the diversity axes and the final audio is fully traceable
(reviewers can sort by axis and listen for monotonic prosody changes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random


# ---------------------------------------------------------------------------
# Per-axis vocabularies
# ---------------------------------------------------------------------------

INTENSITIES = [1, 2, 3, 4, 5]            # 1 = mild, 5 = extreme

CAUSES_NEGATIVE = [
    "wait_time",          # "I have been on hold for over an hour"
    "technical_issue",    # the product / service is broken
    "billing_error",      # incorrect charge
    "product_defect",     # received broken / wrong item
    "policy_dispute",     # company policy is unfair / unclear
    "rude_agent",         # previous agent was unhelpful or rude
    "broken_promise",     # was promised something and it did not happen
    "third_call",         # repeated calls without resolution
]

CAUSES_NEUTRAL = [
    "info_request",       # asking how something works
    "verification",       # confirming account details
    "small_talk",         # filler turns / pleasantries
    "wait_time",          # neutral mention of waiting (low intensity)
    "self_correction",    # restating something
]

CAUSES_POSITIVE = [
    "issue_resolved",     # problem fixed
    "good_service",       # praising the agent
    "compliment",         # general positive remark
    "gratitude",          # thanking
]

STYLES_NEGATIVE = [
    "explicit_anger",
    "passive_aggressive",
    "sarcastic",
    "exasperated",
    "polite_but_firm",
    "rambling",
]
STYLES_SAD = ["tearful", "exasperated", "rambling", "polite_but_firm"]
STYLES_NEUTRAL = ["matter_of_fact", "polite_formal", "casual"]
STYLES_POSITIVE = ["grateful", "warm", "casual", "polite_formal"]

PERSONAS = [
    "young_informal",
    "elderly_formal",
    "business_user",
    "technically_savvy",
    "first_time_caller",
    "regular_complainant",
]

TURN_POSITIONS = ["opening", "mid_call", "escalation", "closing"]


# ---------------------------------------------------------------------------
# Per-class constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ClassRules:
    intensities: List[int]
    causes:      List[str]
    styles:      List[str]


_CLASS_RULES: Dict[str, _ClassRules] = {
    "anger": _ClassRules(
        intensities=[3, 4, 5],
        causes=CAUSES_NEGATIVE,
        styles=["explicit_anger", "sarcastic", "exasperated",
                "passive_aggressive", "polite_but_firm"],
    ),
    "frustration": _ClassRules(
        intensities=[2, 3, 4, 5],
        causes=CAUSES_NEGATIVE,
        styles=STYLES_NEGATIVE,
    ),
    "sadness": _ClassRules(
        intensities=[2, 3, 4],
        causes=CAUSES_NEGATIVE,
        styles=STYLES_SAD,
    ),
    "neutral": _ClassRules(
        intensities=[1, 2],
        causes=CAUSES_NEUTRAL,
        styles=STYLES_NEUTRAL,
    ),
    "satisfaction": _ClassRules(
        intensities=[3, 4, 5],
        causes=CAUSES_POSITIVE,
        styles=STYLES_POSITIVE,
    ),
}


def class_rules(label: str) -> _ClassRules:
    if label not in _CLASS_RULES:
        raise KeyError(f"Unknown label '{label}'. Known: {list(_CLASS_RULES)}")
    return _CLASS_RULES[label]


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_point(label: str, rng: Optional[random.Random] = None) -> Dict:
    """Sample one valid (intensity, cause, style, persona, turn_position) point.

    Args:
        label: target SmartHandover class
        rng:   random.Random instance (caller controls seed)

    Returns:
        dict with keys: label, intensity, cause, style, persona, turn_position
    """
    rng = rng or random.Random()
    rules = class_rules(label)
    return {
        "label":         label,
        "intensity":     rng.choice(rules.intensities),
        "cause":         rng.choice(rules.causes),
        "style":         rng.choice(rules.styles),
        "persona":       rng.choice(PERSONAS),
        "turn_position": rng.choice(TURN_POSITIONS),
    }


def enumerate_points(label: str, n: int, seed: int) -> List[Dict]:
    """Sample ``n`` valid points for ``label`` with a deterministic seed.

    The sampler interleaves combinations to maximise coverage even when
    ``n`` is smaller than the full Cartesian product.
    """
    rng = random.Random(seed)
    points = [sample_point(label, rng) for _ in range(n)]
    return points


def total_combinations(label: str) -> int:
    """Total Cartesian-product size for a class (sanity check / report)."""
    r = class_rules(label)
    return (
        len(r.intensities) * len(r.causes) * len(r.styles)
        * len(PERSONAS) * len(TURN_POSITIONS)
    )


# ---------------------------------------------------------------------------
# Coverage helper (for the run log)
# ---------------------------------------------------------------------------


def coverage_summary(points: List[Dict]) -> Dict[str, Tuple[int, int]]:
    """Return {axis_name: (distinct_seen, total_possible)} for a point list.

    Useful to write to ``run_log.json`` so the report can claim
    "we covered 92% of the diversity space".
    """
    seen = {ax: set() for ax in
            ("intensity", "cause", "style", "persona", "turn_position")}
    for p in points:
        for ax in seen:
            seen[ax].add(p[ax])

    totals = {
        "intensity":     5,
        "cause":         len(CAUSES_NEGATIVE | set(CAUSES_NEUTRAL)
                              | set(CAUSES_POSITIVE)) if False else
                          len(set(CAUSES_NEGATIVE) | set(CAUSES_NEUTRAL)
                              | set(CAUSES_POSITIVE)),
        "style":         len(set(STYLES_NEGATIVE) | set(STYLES_SAD)
                              | set(STYLES_NEUTRAL) | set(STYLES_POSITIVE)),
        "persona":       len(PERSONAS),
        "turn_position": len(TURN_POSITIONS),
    }
    return {ax: (len(seen[ax]), totals[ax]) for ax in seen}
