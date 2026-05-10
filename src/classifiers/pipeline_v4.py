"""SmartHandover - End-to-End Inference Pipeline v4.

audio (16 kHz mono) ->
  Whisper ASR              (transcription)
  VADER + GoEmo + RoBERTa  (text scores)
  Wav2Vec2 fine-tuned       (5 native audio scores)
  -> concat into 20-dim feature vector
  -> meta_classifier_v4_calibrated.pkl (LR + isotonic, 5-class)
  -> handover_score = w_anger * P(anger) + w_frust * P(frust)
  -> compare to threshold (handover_threshold_v4.json)

The text components (Whisper, VADER, GoEmo, RoBERTa) are inherited
from the v1 pipeline (``src.classifiers.pipeline``); this module only
swaps the audio classifier and the meta-classifier.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, Optional

import joblib
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.classifiers.goemo_classifier import GoEmotionsClassifier  # noqa: E402
from src.classifiers.pipeline import RobertaTextClassifier  # noqa: E402
from src.classifiers.vader_classifier import VaderClassifier  # noqa: E402
from src.classifiers.wav2vec2_finetuned import (  # noqa: E402
    Wav2Vec2FineTunedClassifier,
)
from src.classifiers.whisper_asr import WhisperASR  # noqa: E402

CKPT_DIR = "checkpoints"
DATA_DIR = os.path.join("data", "processed")
CONFIG_DIR = "configs"

TARGET_LABELS = ["anger", "frustration", "sadness", "neutral", "satisfaction"]

# v4 feature schema (20 dims: 5 RoBERTa + 6 GoEmo + 4 VADER + 5 audio)
ROBERTA_COLS = ["prob_anger", "prob_frust", "prob_sadne",
                  "prob_neutr", "prob_satis"]
GOEMO_COLS  = ["goemo_anger", "goemo_disgust", "goemo_fear",
                "goemo_joy", "goemo_neutral", "goemo_sadness"]
VADER_COLS  = ["vader_pos", "vader_neg", "vader_neu", "vader_compound"]
AUDIO_COLS_V4 = ["audio_anger", "audio_frust", "audio_sad",
                  "audio_neut", "audio_satis"]
FEATURE_COLUMNS_V4 = ROBERTA_COLS + GOEMO_COLS + VADER_COLS + AUDIO_COLS_V4

DEFAULT_CONFIG: Dict[str, Any] = {
    "whisper_size":       "small",
    "roberta_checkpoint": os.path.join(CKPT_DIR, "roberta_combined.pt"),
    "audio_checkpoint":   os.path.join(CKPT_DIR, "wav2vec2_finetuned.pt"),
    "meta_checkpoint":    os.path.join(CKPT_DIR,
                                          "meta_classifier_v4_calibrated.pkl"),
    "threshold_json":     os.path.join(CONFIG_DIR,
                                          "handover_threshold_v4.json"),
    "device":             None,
    "max_text_length":    128,
    "fallback_threshold": 0.20,
    "fallback_w_anger":   0.6,
    "fallback_w_frust":   0.4,
}


class SmartHandoverPipelineV4:
    """v4 audio -> handover decision pipeline (best-of state).

    Differences vs v1:
      * RoBERTa: ``roberta_combined.pt`` (multi-corpus, MELD + synth).
      * Audio: ``wav2vec2_finetuned.pt`` (Day F13, 5 native classes).
      * Meta: ``meta_classifier_v4_calibrated.pkl`` (LR + isotonic, 20-dim).
      * Handover: weighted score (w_anger * P(anger) + w_frust * P(frust))
                  vs single threshold ``configs/handover_threshold_v4.json``.
    """

    _ANGER_IDX = TARGET_LABELS.index("anger")
    _FRUST_IDX = TARGET_LABELS.index("frustration")

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg: Dict[str, Any] = dict(DEFAULT_CONFIG)
        if config:
            cfg.update(config)
        self.config = cfg

        device = cfg["device"] or ("cuda" if torch.cuda.is_available()
                                    else "cpu")
        self.device = device
        print(f"[SmartHandoverPipelineV4] Initialising on device={device}")

        # --- 1. Whisper ASR ---
        print("  loading Whisper ...")
        self.whisper = WhisperASR(model_size=cfg["whisper_size"],
                                    device=device)

        # --- 2. VADER ---
        self.vader = VaderClassifier()

        # --- 3. GoEmotions ---
        print("  loading GoEmotions ...")
        self.goemo = GoEmotionsClassifier(
            device=(0 if device == "cuda" else -1),
        )

        # --- 4. RoBERTa fine-tuned (multi-corpus combined) ---
        print(f"  loading RoBERTa (combined): {cfg['roberta_checkpoint']}")
        self.roberta = RobertaTextClassifier(
            ckpt_path=cfg["roberta_checkpoint"],
            device=device, max_length=cfg["max_text_length"],
        )

        # --- 5. Wav2Vec2 fine-tuned (5-class native) ---
        print(f"  loading Wav2Vec2 fine-tuned: {cfg['audio_checkpoint']}")
        self.audio = Wav2Vec2FineTunedClassifier(
            checkpoint_path=cfg["audio_checkpoint"], device=device,
        )

        # --- 6. Meta-classifier v4 ---
        print(f"  loading meta v4: {cfg['meta_checkpoint']}")
        bundle = joblib.load(cfg["meta_checkpoint"])
        self.meta = bundle["model"]
        self.meta_name = bundle.get("name", type(self.meta).__name__)
        self.feature_columns = bundle.get("feature_columns",
                                            FEATURE_COLUMNS_V4)

        # --- 7. Threshold + weights ---
        self._load_threshold(cfg)
        print(f"  threshold = {self.threshold:.3f}, "
              f"w_anger={self.w_anger}, w_frust={self.w_frust}")
        print("[SmartHandoverPipelineV4] Ready.\n")

    def _load_threshold(self, cfg: Dict[str, Any]) -> None:
        path = cfg["threshold_json"]
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.threshold = float(data["optimal_threshold"])
                self.w_anger   = float(data["weights"]["w_anger"])
                self.w_frust   = float(data["weights"]["w_frust"])
                return
            except (KeyError, ValueError, OSError):
                pass
        self.threshold = float(cfg["fallback_threshold"])
        self.w_anger   = float(cfg["fallback_w_anger"])
        self.w_frust   = float(cfg["fallback_w_frust"])

    # -----------------------------------------------------------------
    # Feature vector assembly (20 dim)
    # -----------------------------------------------------------------

    def _build_feature_vector(
        self,
        vader_scores: Dict[str, float],
        goemo_scores: Dict[str, float],
        roberta_scores: Dict[str, float],
        audio_scores: Dict[str, float],
    ) -> np.ndarray:
        lookup: Dict[str, float] = {}
        # VADER
        lookup.update({
            "vader_pos":      vader_scores["pos"],
            "vader_neg":      vader_scores["neg"],
            "vader_neu":      vader_scores["neu"],
            "vader_compound": vader_scores["compound"],
        })
        # GoEmo (6)
        for key in GOEMO_COLS:
            raw = key.replace("goemo_", "")
            lookup[key] = float(goemo_scores.get(raw, 0.0))
        # RoBERTa (5)
        for key in ROBERTA_COLS:
            lookup[key] = float(roberta_scores.get(key, 0.0))
        # Audio v4 (5 native)
        lookup["audio_anger"] = float(audio_scores.get("anger",        0.0))
        lookup["audio_frust"] = float(audio_scores.get("frustration",  0.0))
        lookup["audio_sad"]   = float(audio_scores.get("sadness",      0.0))
        lookup["audio_neut"]  = float(audio_scores.get("neutral",      0.0))
        lookup["audio_satis"] = float(audio_scores.get("satisfaction", 0.0))
        return np.array([lookup[c] for c in self.feature_columns],
                         dtype=np.float32).reshape(1, -1)

    # -----------------------------------------------------------------
    # Main inference call
    # -----------------------------------------------------------------

    def predict_from_audio(
        self, audio_array: np.ndarray, sr: int = 16000,
        text: Optional[str] = None,
    ) -> Dict[str, Any]:
        audio_array = np.asarray(audio_array, dtype=np.float32).flatten()
        timings: Dict[str, float] = {}

        t = time.perf_counter()
        if text is None:
            text = self.whisper.transcribe(audio_array, sr=sr)
        timings["asr"] = time.perf_counter() - t

        t = time.perf_counter()
        vader_scores = self.vader.predict(text)
        timings["vader"] = time.perf_counter() - t

        t = time.perf_counter()
        goemo_scores = self.goemo.predict(text)
        timings["goemo"] = time.perf_counter() - t

        t = time.perf_counter()
        roberta_scores = self.roberta.predict(text)
        timings["roberta"] = time.perf_counter() - t

        t = time.perf_counter()
        audio_scores = self.audio.predict(audio_array, sr=sr)
        timings["audio_v4"] = time.perf_counter() - t

        t = time.perf_counter()
        feat = self._build_feature_vector(
            vader_scores, goemo_scores, roberta_scores, audio_scores,
        )
        if hasattr(self.meta, "predict_proba"):
            meta_probs = self.meta.predict_proba(feat)[0]
        else:
            preds = int(self.meta.predict(feat)[0])
            meta_probs = np.zeros(len(TARGET_LABELS), dtype=np.float32)
            meta_probs[preds] = 1.0
        pred_idx = int(np.argmax(meta_probs))
        timings["meta"] = time.perf_counter() - t

        # v4 weighted handover score
        handover_score = float(
            self.w_anger * meta_probs[self._ANGER_IDX]
            + self.w_frust * meta_probs[self._FRUST_IDX]
        )
        should_handover = bool(handover_score > self.threshold)

        timings["total"] = sum(timings.values())

        return {
            "text":              text,
            "predicted_emotion": TARGET_LABELS[pred_idx],
            "confidence":        float(meta_probs[pred_idx]),
            "should_handover":   should_handover,
            "handover_score":    handover_score,
            "threshold":         self.threshold,
            "w_anger":           self.w_anger,
            "w_frust":           self.w_frust,
            "meta_probs":        {TARGET_LABELS[i]: float(meta_probs[i])
                                  for i in range(len(TARGET_LABELS))},
            "timings":           timings,
            "details": {
                "vader":          vader_scores,
                "goemo":          goemo_scores,
                "roberta":        roberta_scores,
                "audio_v4":       audio_scores,  # 5 native classes
            },
        }
