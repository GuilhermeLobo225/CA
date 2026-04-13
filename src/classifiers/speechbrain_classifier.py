"""
SmartHandover — Audio Emotion Classifier (wav2vec2, IEMOCAP)
Uses superb/wav2vec2-large-superb-er via HuggingFace transformers.

Fallback from SpeechBrain native interface due to Windows symlink/k2
compatibility issues. Same IEMOCAP 4-class output.

IEMOCAP classes: ang (angry), hap (happy), sad (sad), neu (neutral)

Mapping to SmartHandover target classes:
    ang -> anger  (also used as frustration proxy in ensemble)
    hap -> satisfaction
    sad -> sadness
    neu -> neutral
"""

import torch
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# IEMOCAP label order as returned by the superb/wav2vec2-large-superb-er model
# model.config.id2label = {0: 'neu', 1: 'hap', 2: 'ang', 3: 'sad'}
IEMOCAP_LABELS = ["neu", "hap", "ang", "sad"]

# Mapping to our 5 target classes (argmax-based)
IEMOCAP_TO_TARGET = {
    "ang": "anger",
    "hap": "satisfaction",
    "sad": "sadness",
    "neu": "neutral",
}

MODEL_ID = "superb/wav2vec2-large-superb-er"


class SpeechBrainClassifier:
    """Audio emotion classifier using wav2vec2 trained on IEMOCAP.

    Despite the class name (kept for compatibility with the rest of the
    codebase), this uses HuggingFace transformers instead of SpeechBrain.
    """

    def __init__(self, device: str = "cpu"):
        """Load the pre-trained model.

        Args:
            device: 'cpu' or 'cuda'.
        """
        self.device = device
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)
        self.model.to(device)
        self.model.eval()

        # Build id->label from model config
        self.id2label = self.model.config.id2label  # {0:'neu', 1:'hap', 2:'ang', 3:'sad'}

    @torch.no_grad()
    def predict(self, audio_array, sr: int = 16000) -> dict:
        """Classify emotion from a raw audio waveform.

        Args:
            audio_array: 1-D NumPy array or torch Tensor (float32, mono, 16kHz).
            sr: Sampling rate (must be 16000).

        Returns:
            Dict mapping IEMOCAP labels to probabilities:
            {"ang": float, "hap": float, "sad": float, "neu": float}
        """
        if isinstance(audio_array, torch.Tensor):
            audio_array = audio_array.cpu().numpy()

        audio_array = audio_array.astype(np.float32).flatten()

        inputs = self.feature_extractor(
            audio_array, sampling_rate=16000, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logits = self.model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        return {self.id2label[i]: float(probs[i]) for i in range(len(probs))}

    def map_to_target_class(self, speechbrain_probs: dict) -> str:
        """Map IEMOCAP probabilities to a SmartHandover target class.

        Uses argmax: ang->anger, hap->satisfaction, sad->sadness, neu->neutral.

        Args:
            speechbrain_probs: Dict of IEMOCAP label probabilities.

        Returns:
            Target class name.
        """
        best_label = max(speechbrain_probs, key=speechbrain_probs.get)
        return IEMOCAP_TO_TARGET[best_label]

    def predict_and_classify(self, audio_array, sr: int = 16000) -> dict:
        """Convenience: predict probs and map to target class."""
        probs = self.predict(audio_array, sr)
        probs["predicted_class"] = self.map_to_target_class(probs)
        return probs
