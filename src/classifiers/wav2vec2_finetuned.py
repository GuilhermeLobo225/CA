"""SmartHandover - Inference wrapper for the fine-tuned wav2vec2 (v4).

Loads ``checkpoints/wav2vec2_finetuned.pt`` (Day F13) and exposes a
``predict(audio_array, sr)`` method that returns the 5 native target
probabilities (anger, frustration, sadness, neutral, satisfaction).

Used by the v4 pipeline (``pipeline_v4.py``) and the Gradio demo.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import torch

from src.training.train_audio import (  # noqa: E402
    NUM_CLASSES, build_model,
)


CKPT_DIR = "checkpoints"
DEFAULT_CHECKPOINT = os.path.join(CKPT_DIR, "wav2vec2_finetuned.pt")
TARGET_LABELS_5 = ["anger", "frustration", "sadness", "neutral", "satisfaction"]


class Wav2Vec2FineTunedClassifier:
    """Drop-in inference wrapper around the Day-F13 fine-tuned wav2vec2.

    Same surface as SpeechBrainClassifier (``predict`` returns a dict),
    but the dict has 5 keys (the 5-class native output) instead of the
    4 IEMOCAP labels. Use ``predict_native`` for the same dict and
    ``predict_iemocap_collapsed`` if you need the legacy 4-label form.
    """

    def __init__(self, checkpoint_path: Optional[str] = None,
                 device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "cpu")
        self.checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Wav2Vec2 fine-tuned checkpoint not found: "
                f"{self.checkpoint_path}. Run Day F13 first."
            )
        self.model, self.feature_extractor = build_model(
            num_classes=NUM_CLASSES, device=self.device,
        )
        self.model.load_state_dict(
            torch.load(self.checkpoint_path,
                        map_location=self.device, weights_only=True)
        )
        self.model.eval()

    @torch.no_grad()
    def predict(self, audio_array, sr: int = 16000) -> Dict[str, float]:
        """Return the 5 native target-label probabilities."""
        if isinstance(audio_array, torch.Tensor):
            audio_array = audio_array.cpu().numpy()
        audio_array = np.asarray(audio_array, dtype=np.float32).flatten()

        # Cap the clip at 8 s as in training
        max_samples = int(8.0 * sr)
        if len(audio_array) > max_samples:
            start = (len(audio_array) - max_samples) // 2
            audio_array = audio_array[start:start + max_samples]

        # Zero-mean unit-RMS normalisation (training-time setting)
        audio_array = audio_array - np.mean(audio_array)
        rms = float(np.sqrt(np.mean(audio_array ** 2) + 1e-12))
        if rms > 0:
            audio_array = audio_array / rms * 0.1

        inputs = self.feature_extractor(
            audio_array, sampling_rate=sr, return_tensors="pt", padding=False,
        )
        input_values = inputs["input_values"].to(self.device)
        attn = inputs.get("attention_mask")
        if attn is not None:
            attn = attn.to(self.device)

        with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
            out = self.model(input_values=input_values,
                              attention_mask=attn)
        probs = torch.softmax(out.logits, dim=-1).squeeze(0).cpu().numpy()
        return {label: float(probs[i])
                for i, label in enumerate(TARGET_LABELS_5)}

    def predict_iemocap_collapsed(self, audio_array,
                                   sr: int = 16000) -> Dict[str, float]:
        """Return 4 IEMOCAP-style probs (ang/hap/sad/neu) by collapsing
        the 5-class output. Used for backwards compatibility with the
        v1/v2/v3 19-dim feature schema:
            ang = anger + frustration
            hap = satisfaction
            sad = sadness
            neu = neutral
        """
        p = self.predict(audio_array, sr=sr)
        return {
            "ang": p["anger"] + p["frustration"],
            "hap": p["satisfaction"],
            "sad": p["sadness"],
            "neu": p["neutral"],
        }
