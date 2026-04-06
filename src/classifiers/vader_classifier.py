"""
SmartHandover — VADER Sentiment Classifier
Lexicon-based baseline using VADER (Valence Aware Dictionary and sEntiment Reasoner).

Maps compound scores to the 5 target classes:
  compound < -0.3           -> frustration/anger  (mapped to "anger")
  -0.3 <= compound <= 0.1   -> sadness/neutral    (mapped to "sadness" if <= -0.05, else "neutral")
  0.1 < compound <= 0.3     -> neutral
  compound > 0.3            -> satisfaction
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class VaderClassifier:
    """Wrapper around VADER for emotion classification."""

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def predict(self, text: str) -> dict:
        """Analyse sentiment of a text string.

        Args:
            text: Input text to analyse.

        Returns:
            Dictionary with keys: pos, neg, neu, compound (all floats).
        """
        scores = self.analyzer.polarity_scores(text)
        return {
            "pos": scores["pos"],
            "neg": scores["neg"],
            "neu": scores["neu"],
            "compound": scores["compound"],
        }

    def map_to_target_class(self, compound_score: float) -> str:
        """Map a VADER compound score to one of the 5 SmartHandover target classes.

        Mapping rules:
            compound < -0.3                -> "anger"  (frustration/anger zone)
            -0.3 <= compound <= -0.05      -> "sadness"
            -0.05 < compound <= 0.1        -> "neutral"
            0.1 < compound <= 0.3          -> "neutral"
            compound > 0.3                 -> "satisfaction"

        Args:
            compound_score: VADER compound score in [-1, 1].

        Returns:
            One of: "anger", "frustration", "sadness", "neutral", "satisfaction".
        """
        if compound_score < -0.3:
            return "anger"
        elif compound_score <= -0.05:
            return "sadness"
        elif compound_score <= 0.1:
            return "neutral"
        elif compound_score <= 0.3:
            return "neutral"
        else:
            return "satisfaction"

    def predict_and_classify(self, text: str) -> dict:
        """Convenience method: predict scores and map to target class.

        Returns:
            Dictionary with VADER scores + 'predicted_class'.
        """
        scores = self.predict(text)
        scores["predicted_class"] = self.map_to_target_class(scores["compound"])
        return scores
