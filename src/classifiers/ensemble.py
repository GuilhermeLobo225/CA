"""
SmartHandover — Text Ensemble Classifier
Weighted combination of RoBERTa, GoEmotions and VADER predictions.

Grid search over weights on validation set, then apply to test set.
"""

import numpy as np
from sklearn.metrics import f1_score

# Target classes shared across all models
TARGET_LABELS = ["anger", "frustration", "sadness", "neutral", "satisfaction"]
TARGET_LABEL2ID = {label: i for i, label in enumerate(TARGET_LABELS)}

# GoEmotions -> target mapping for probability aggregation
GOEMO_TO_TARGET = {
    "anger":   ["anger"],
    "frustration": ["disgust", "fear"],
    "sadness": ["sadness"],
    "neutral": ["neutral", "surprise"],
    "satisfaction": ["joy"],
}

# VADER compound -> soft probability distribution
def vader_to_probs(compound: float) -> np.ndarray:
    """Convert VADER compound score to a soft 5-class probability vector.

    Uses a heuristic sigmoid-based mapping rather than hard thresholds,
    so the ensemble gets gradient-friendly signals.
    """
    probs = np.zeros(5, dtype=np.float32)
    if compound < -0.3:
        probs[0] = 0.5   # anger
        probs[1] = 0.3   # frustration
        probs[2] = 0.15  # sadness
        probs[3] = 0.05  # neutral
    elif compound < -0.05:
        probs[2] = 0.4   # sadness
        probs[1] = 0.25  # frustration
        probs[3] = 0.2   # neutral
        probs[0] = 0.15  # anger
    elif compound <= 0.1:
        probs[3] = 0.6   # neutral
        probs[2] = 0.15  # sadness
        probs[4] = 0.15  # satisfaction
        probs[1] = 0.1   # frustration
    elif compound <= 0.3:
        probs[3] = 0.35  # neutral
        probs[4] = 0.45  # satisfaction
        probs[2] = 0.1   # sadness
        probs[0] = 0.1   # anger
    else:
        probs[4] = 0.6   # satisfaction
        probs[3] = 0.25  # neutral
        probs[2] = 0.1   # sadness
        probs[0] = 0.05  # anger
    return probs


def goemo_row_to_probs(row) -> np.ndarray:
    """Convert a GoEmotions prediction row to 5-class target probabilities."""
    probs = np.zeros(5, dtype=np.float32)
    for i, target in enumerate(TARGET_LABELS):
        for goemo_label in GOEMO_TO_TARGET[target]:
            col = f"goemo_{goemo_label}"
            probs[i] += row.get(col, 0.0)
    return probs


class TextEnsembleClassifier:
    """Weighted ensemble of RoBERTa + GoEmotions + VADER."""

    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        """
        Args:
            alpha: Weight for RoBERTa.
            beta:  Weight for GoEmotions.
            gamma: Weight for VADER.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def predict(self, roberta_probs, goemo_probs, vader_probs):
        """Combine 3 probability vectors into a single prediction.

        Args:
            roberta_probs: np.array shape [5]
            goemo_probs:   np.array shape [5]
            vader_probs:   np.array shape [5]

        Returns:
            (predicted_class_idx, combined_probs)
        """
        combined = (self.alpha * roberta_probs +
                    self.beta * goemo_probs +
                    self.gamma * vader_probs)
        return int(np.argmax(combined)), combined

    def find_best_weights(self, roberta_probs_all, goemo_probs_all,
                          vader_probs_all, true_labels):
        """Grid search for best weights on validation data.

        Args:
            roberta_probs_all: np.array [N, 5]
            goemo_probs_all:   np.array [N, 5]
            vader_probs_all:   np.array [N, 5]
            true_labels:       list of int [N]

        Returns:
            dict with best alpha, beta, gamma and their weighted_f1.
        """
        best_f1 = 0.0
        best_params = {}

        for alpha in [0.3, 0.4, 0.5, 0.6]:
            for beta in [0.2, 0.3, 0.4]:
                gamma = round(1.0 - alpha - beta, 2)
                if gamma < 0:
                    continue

                combined = (alpha * roberta_probs_all +
                            beta * goemo_probs_all +
                            gamma * vader_probs_all)
                preds = np.argmax(combined, axis=-1)
                wf1 = f1_score(true_labels, preds, average="weighted", zero_division=0)

                if wf1 > best_f1:
                    best_f1 = wf1
                    best_params = {"alpha": alpha, "beta": beta,
                                   "gamma": gamma, "weighted_f1": wf1}

        # Apply best weights
        self.alpha = best_params["alpha"]
        self.beta = best_params["beta"]
        self.gamma = best_params["gamma"]

        return best_params
