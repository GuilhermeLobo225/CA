"""
SmartHandover — GoEmotions Zero-Shot Classifier
Wrapper around j-hartmann/emotion-english-distilroberta-base.

Original model outputs 7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise.

Mapping to SmartHandover target classes (via probability aggregation):
    anger       <- anger
    frustration <- disgust + fear  (proxies)
    sadness     <- sadness
    neutral     <- neutral + surprise
    satisfaction <- joy
"""

from transformers import pipeline


# GoEmotions labels as returned by the model
GOEMO_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# Mapping: target class -> list of GoEmotions labels whose probs are summed
TARGET_MAPPING = {
    "anger":        ["anger"],
    "frustration":  ["disgust", "fear"],
    "sadness":      ["sadness"],
    "neutral":      ["neutral", "surprise"],
    "satisfaction": ["joy"],
}


class GoEmotionsClassifier:
    """Zero-shot emotion classifier using a pre-trained DistilRoBERTa model."""

    def __init__(self, device: int = -1, batch_size: int = 32):
        """Load the HuggingFace pipeline.

        Args:
            device: -1 for CPU, 0+ for GPU index.
            batch_size: Default batch size for predict_batch.
        """
        self.batch_size = batch_size
        self.pipe = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,   # return all 7 emotion scores
            device=device,
        )

    def predict(self, text: str) -> dict:
        """Classify a single text.

        Args:
            text: Input text.

        Returns:
            Dict mapping each of the 7 GoEmotions labels to its probability.
        """
        results = self.pipe(text)[0]  # list of {label, score} dicts
        return {item["label"]: item["score"] for item in results}

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Classify a batch of texts efficiently.

        Args:
            texts: List of input texts.

        Returns:
            List of dicts, each mapping GoEmotions labels to probabilities.
        """
        all_results = self.pipe(texts, batch_size=self.batch_size)
        return [
            {item["label"]: item["score"] for item in sample}
            for sample in all_results
        ]

    def map_to_target_class(self, goemo_probs: dict) -> str:
        """Aggregate GoEmotions probabilities into 5 target classes and pick the best.

        Aggregation rules:
            anger        = P(anger)
            frustration  = P(disgust) + P(fear)
            sadness      = P(sadness)
            neutral      = P(neutral) + P(surprise)
            satisfaction = P(joy)

        Args:
            goemo_probs: Dict of 7 GoEmotions label probabilities.

        Returns:
            Target class name with the highest aggregated probability.
        """
        target_scores = {}
        for target, source_labels in TARGET_MAPPING.items():
            target_scores[target] = sum(goemo_probs.get(l, 0.0) for l in source_labels)
        return max(target_scores, key=target_scores.get)

    def predict_and_classify(self, text: str) -> dict:
        """Convenience: predict raw scores and map to target class.

        Returns:
            Dict with all 7 GoEmotions probs + 'predicted_class'.
        """
        probs = self.predict(text)
        probs["predicted_class"] = self.map_to_target_class(probs)
        return probs
