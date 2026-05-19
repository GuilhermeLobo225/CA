"""Soft-vote stacking ensemble used by the v5 meta-classifier.

A standalone module so joblib can pickle / unpickle ``StackingSoftVote``
instances across scripts.
"""

from __future__ import annotations

import numpy as np


class StackingSoftVote:
    """Average ``predict_proba`` across a list of fitted sklearn-style models."""

    def __init__(self, models):
        self.models = list(models)

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
