"""
SmartHandover — Handover Decision Logic (Day 10)

Two complementary rules turn a stream of per-utterance emotion probabilities
into a binary "hand the caller over to a human" decision:

  Rule 1 — Instantaneous strong negative emotion.
      The latest utterance alone carries enough negative signal
          P(anger) + P(frustration) > threshold
      to justify an immediate handover.

  Rule 2 — Sustained negative trend.
      The rolling window of the last ``window_size`` utterances has an
      average negative mass
          mean( P(anger) + P(frustration) + P(sadness) ) > threshold * 0.7
      indicating the caller has been progressively deteriorating.

The class exposes:

    update(prediction_dict)
    should_handover()  -> (bool, reason_string)

A ``prediction_dict`` is expected to contain either the 5 probability keys
directly (anger / frustration / sadness / neutral / satisfaction) or a
``meta_probs`` nested dict (as produced by ``SmartHandoverPipeline``).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterable, Optional, Tuple


# The 5 target classes, in canonical index order
_TARGETS = ("anger", "frustration", "sadness", "neutral", "satisfaction")


def _extract_probs(pred: Dict[str, Any]) -> Dict[str, float]:
    """Normalise an incoming prediction into {label -> prob}.

    Accepts either::

        {"anger": 0.2, "frustration": 0.6, ...}

    or::

        {"predicted_emotion": "frustration",
         "meta_probs": {"anger": 0.2, "frustration": 0.6, ...},
         ...}
    """
    if "meta_probs" in pred and isinstance(pred["meta_probs"], dict):
        probs = pred["meta_probs"]
    else:
        probs = pred

    out: Dict[str, float] = {}
    for t in _TARGETS:
        out[t] = float(probs.get(t, 0.0))
    return out


class HandoverDecision:
    """Stateful handover decision with a sliding window.

    Args:
        threshold: Base probability threshold. Rule 1 triggers when the
            latest ``P(anger)+P(frustration) > threshold``. Rule 2 uses
            ``threshold * 0.7`` as its softer rolling-average threshold.
        window_size: How many past predictions Rule 2 averages over.

    Typical use::

        dec = HandoverDecision(threshold=0.3, window_size=3)
        for prediction in stream:
            dec.update(prediction)
            triggered, reason = dec.should_handover()
            if triggered:
                route_to_agent(reason)
                break
    """

    #: Reason strings — stable API for downstream consumers
    REASON_INSTANT = "instant_strong_emotion"
    REASON_TREND   = "sustained_negative_trend"
    REASON_OK      = "ok"

    def __init__(self, threshold: float = 0.6, window_size: int = 3):
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")

        self.threshold = float(threshold)
        self.window_size = int(window_size)
        # deque auto-drops elements older than window_size
        self.history: deque[Dict[str, float]] = deque(maxlen=self.window_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, prediction_dict: Dict[str, Any]) -> None:
        """Append a new prediction to the history buffer."""
        self.history.append(_extract_probs(prediction_dict))

    def reset(self) -> None:
        """Clear the sliding window (start a new conversation)."""
        self.history.clear()

    def should_handover(self) -> Tuple[bool, str]:
        """Evaluate the two rules against the current history.

        Returns:
            ``(True, reason)``  if either rule fires.
            ``(False, "ok")``   otherwise (including when history is empty).
        """
        if not self.history:
            return False, self.REASON_OK

        # ---- Rule 1: instantaneous strong emotion ------------------
        latest = self.history[-1]
        instant = latest["anger"] + latest["frustration"]
        if instant > self.threshold:
            return True, self.REASON_INSTANT

        # ---- Rule 2: sustained negative trend ----------------------
        # Only fire once the window is full — partial windows would be
        # noisy in early conversation.
        if len(self.history) >= self.window_size:
            negative_mass = sum(
                h["anger"] + h["frustration"] + h["sadness"]
                for h in self.history
            ) / len(self.history)
            if negative_mass > self.threshold * 0.7:
                return True, self.REASON_TREND

        return False, self.REASON_OK

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def process_stream(
        self, predictions: Iterable[Dict[str, Any]],
    ) -> Optional[Tuple[int, str]]:
        """Feed a whole conversation and return the first handover event.

        Args:
            predictions: Iterable of prediction dicts in chronological order.

        Returns:
            ``(utterance_index, reason)`` of the first trigger, or None if
            no rule fired for the entire conversation. The buffer is
            reset before processing so it is safe to re-use the same
            instance for several calls.
        """
        self.reset()
        for idx, pred in enumerate(predictions):
            self.update(pred)
            triggered, reason = self.should_handover()
            if triggered:
                return idx, reason
        return None

    def __repr__(self) -> str:  # pragma: no cover
        return (f"HandoverDecision(threshold={self.threshold:.3f}, "
                f"window_size={self.window_size}, "
                f"history_len={len(self.history)})")
