"""
SmartHandover — End-to-End Inference Pipeline (Day 9)

``SmartHandoverPipeline`` glues together every component required to turn a
raw audio array into a handover decision:

    audio (float32, 16 kHz)
      -> Whisper ASR           (transcription)
      -> VADER + GoEmo + RoBERTa  (text scores)
      -> SpeechBrain           (audio scores)
      -> concat -> 19-dim feature
      -> Meta-classifier       (5-class emotion)
      -> thresholded P(anger)+P(frustration) -> handover boolean

The class is constructed from a ``config`` dict so that individual components
can be swapped out in tests.  Any missing configuration keys fall back to
project defaults (same paths used by earlier days).

Quick-test entry point (``python -m src.classifiers.pipeline``) runs the
pipeline on a handful of MELD test-set utterances and reports per-utterance
latency (target < 2 seconds on GPU).
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import torch

# Project imports -----------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.classifiers.ensemble_trainer import (  # noqa: E402
    CKPT_DIR,
    FEATURE_COLUMNS,
    GOEMO_COLS,
    ROBERTA_COLS,
    SB_COLS,
    TARGET_LABELS,
    VADER_COLS,
)
from src.classifiers.goemo_classifier import GoEmotionsClassifier  # noqa: E402
from src.classifiers.speechbrain_classifier import SpeechBrainClassifier  # noqa: E402
from src.classifiers.vader_classifier import VaderClassifier  # noqa: E402
from src.classifiers.whisper_asr import WhisperASR  # noqa: E402

# -- RoBERTa inference wrapper ---------------------------------------------
from transformers import AutoTokenizer  # noqa: E402
from src.training.train_text import TextOnlyClassifier  # noqa: E402

DEFAULT_CONFIG: Dict[str, Any] = {
    "whisper_size": "small",
    "roberta_checkpoint": os.path.join(CKPT_DIR, "roberta_text_only.pt"),
    "meta_checkpoint":    os.path.join(CKPT_DIR, "meta_classifier.pkl"),
    "threshold_json":     os.path.join("configs", "handover_threshold.json"),
    "device": None,             # auto
    "max_text_length": 128,
    "fallback_threshold": 0.5,  # used if threshold_json missing
}


# ---------------------------------------------------------------------------
# Lightweight RoBERTa inference wrapper
# ---------------------------------------------------------------------------

class RobertaTextClassifier:
    """Thin inference wrapper around the fine-tuned TextOnlyClassifier."""

    def __init__(self, ckpt_path: str, device: str = "cpu",
                 max_length: int = 128):
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.model = TextOnlyClassifier()
        self.model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> Dict[str, float]:
        """Return 5-class RoBERTa probabilities for a single utterance."""
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.amp.autocast(
            "cuda", enabled=(self.device == "cuda")
        ):
            logits = self.model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        # Map to the feature column names used by the meta-classifier.
        return {
            "prob_anger":  float(probs[0]),
            "prob_frust":  float(probs[1]),
            "prob_sadne": float(probs[2]),
            "prob_neutr": float(probs[3]),
            "prob_satis": float(probs[4]),
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class SmartHandoverPipeline:
    """Full audio -> emotion -> handover decision pipeline."""

    # Feature-column indices for the handover trigger
    _ANGER_IDX = TARGET_LABELS.index("anger")
    _FRUST_IDX = TARGET_LABELS.index("frustration")

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg: Dict[str, Any] = dict(DEFAULT_CONFIG)
        if config:
            cfg.update(config)
        self.config = cfg

        device = cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        print(f"[SmartHandoverPipeline] Initialising on device={device}")

        # --- 1. Whisper ASR ---
        print("  loading Whisper...")
        self.whisper = WhisperASR(model_size=cfg["whisper_size"], device=device)

        # --- 2. VADER ---
        self.vader = VaderClassifier()

        # --- 3. GoEmotions ---
        print("  loading GoEmotions...")
        self.goemo = GoEmotionsClassifier(
            device=(0 if device == "cuda" else -1),
        )

        # --- 4. SpeechBrain (wav2vec2 IEMOCAP) ---
        print("  loading SpeechBrain audio classifier...")
        self.speechbrain = SpeechBrainClassifier(device=device)

        # --- 5. Fine-tuned RoBERTa ---
        print("  loading RoBERTa text classifier...")
        self.roberta = RobertaTextClassifier(
            ckpt_path=cfg["roberta_checkpoint"],
            device=device,
            max_length=cfg["max_text_length"],
        )

        # --- 6. Meta-classifier ---
        print("  loading meta-classifier...")
        bundle = joblib.load(cfg["meta_checkpoint"])
        self.meta = bundle["model"]
        self.meta_name = bundle.get("name", type(self.meta).__name__)
        self.feature_columns: List[str] = bundle.get(
            "feature_columns", FEATURE_COLUMNS,
        )
        self.target_labels: List[str] = bundle.get(
            "target_labels", TARGET_LABELS,
        )

        # --- 7. Handover threshold (from Day 8) ---
        self.handover_threshold = self._load_threshold(
            cfg["threshold_json"], cfg["fallback_threshold"],
        )
        print(f"  handover threshold = {self.handover_threshold:.3f}")
        print("[SmartHandoverPipeline] Ready.\n")

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _load_threshold(path: str, fallback: float) -> float:
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return float(json.load(f)["optimal_threshold"])
            except (KeyError, ValueError, OSError):
                pass
        print(f"  [WARN] No threshold JSON at {path}, using fallback={fallback}")
        return float(fallback)

    def _build_feature_vector(
        self,
        vader_scores: Dict[str, float],
        goemo_scores: Dict[str, float],
        roberta_scores: Dict[str, float],
        sb_scores: Dict[str, float],
    ) -> np.ndarray:
        """Concatenate the four score dicts into the canonical 19-dim vector."""
        lookup: Dict[str, float] = {}
        # VADER
        lookup.update({
            "vader_pos":      vader_scores["pos"],
            "vader_neg":      vader_scores["neg"],
            "vader_neu":      vader_scores["neu"],
            "vader_compound": vader_scores["compound"],
        })
        # GoEmotions (6 classes used as features)
        for key in GOEMO_COLS:
            raw_key = key.replace("goemo_", "")
            lookup[key] = float(goemo_scores.get(raw_key, 0.0))
        # RoBERTa
        for key in ROBERTA_COLS:
            lookup[key] = float(roberta_scores.get(key, 0.0))
        # SpeechBrain
        lookup.update({
            "sb_ang": float(sb_scores.get("ang", 0.0)),
            "sb_hap": float(sb_scores.get("hap", 0.0)),
            "sb_sad": float(sb_scores.get("sad", 0.0)),
            "sb_neu": float(sb_scores.get("neu", 0.0)),
        })
        vec = np.array(
            [lookup[c] for c in self.feature_columns],
            dtype=np.float32,
        ).reshape(1, -1)
        return vec

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def predict_from_audio(
        self, audio_array: np.ndarray, sr: int = 16000,
        text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the full pipeline on a single audio waveform.

        Args:
            audio_array: 1-D float32 mono waveform at 16 kHz.
            sr: Sampling rate (must be 16000 for Whisper/SpeechBrain).
            text: Optional ground-truth transcript. When provided we skip
                  Whisper (useful for benchmarking latency without ASR).

        Returns:
            Dict with:
                text               : str
                predicted_emotion  : str (one of TARGET_LABELS)
                confidence         : float (max meta prob)
                should_handover    : bool
                handover_score     : float  P(anger)+P(frustration)
                threshold          : float
                meta_probs         : dict[label -> prob]
                timings            : dict[stage -> seconds]
                details            : dict of individual model outputs
        """
        audio_array = np.asarray(audio_array, dtype=np.float32).flatten()
        timings: Dict[str, float] = {}

        # --- a) ASR ---
        t = time.perf_counter()
        if text is None:
            text = self.whisper.transcribe(audio_array, sr=sr)
        timings["asr"] = time.perf_counter() - t

        # --- b) Text models ---
        t = time.perf_counter()
        vader_scores = self.vader.predict(text)
        timings["vader"] = time.perf_counter() - t

        t = time.perf_counter()
        goemo_scores = self.goemo.predict(text)
        timings["goemo"] = time.perf_counter() - t

        t = time.perf_counter()
        roberta_scores = self.roberta.predict(text)
        timings["roberta"] = time.perf_counter() - t

        # --- c) Audio model ---
        t = time.perf_counter()
        sb_scores = self.speechbrain.predict(audio_array, sr=sr)
        timings["speechbrain"] = time.perf_counter() - t

        # --- d) Feature vector + Meta ---
        t = time.perf_counter()
        feat = self._build_feature_vector(
            vader_scores, goemo_scores, roberta_scores, sb_scores,
        )

        if hasattr(self.meta, "predict_proba"):
            meta_probs = self.meta.predict_proba(feat)[0]
        else:
            pred_idx = int(self.meta.predict(feat)[0])
            meta_probs = np.zeros(len(self.target_labels), dtype=np.float32)
            meta_probs[pred_idx] = 1.0
        pred_idx = int(np.argmax(meta_probs))
        timings["meta"] = time.perf_counter() - t

        handover_score = float(
            meta_probs[self._ANGER_IDX] + meta_probs[self._FRUST_IDX]
        )
        should_handover = bool(handover_score > self.handover_threshold)

        total = sum(timings.values())
        timings["total"] = total

        return {
            "text": text,
            "predicted_emotion": self.target_labels[pred_idx],
            "confidence": float(meta_probs[pred_idx]),
            "should_handover": should_handover,
            "handover_score": handover_score,
            "threshold": self.handover_threshold,
            "meta_probs": {
                self.target_labels[i]: float(meta_probs[i])
                for i in range(len(self.target_labels))
            },
            "timings": timings,
            "details": {
                "vader":       vader_scores,
                "goemo":       goemo_scores,
                "roberta":     roberta_scores,
                "speechbrain": sb_scores,
            },
        }


# ===========================================================================
# __main__: smoke test + latency benchmark on MELD test samples
# ===========================================================================


def _safe(s: str, n: int = 70) -> str:
    return str(s)[:n].encode("ascii", errors="replace").decode("ascii")


def _benchmark(pipe: SmartHandoverPipeline, n_samples: int = 5) -> None:
    """Load a handful of MELD test utterances and time the full pipeline."""
    from src.data.load_meld import load_meld

    print(f"[benchmark] Loading {n_samples} MELD test samples ...")
    ds = load_meld(split="test", streaming=False)

    latencies: List[float] = []
    for i in range(min(n_samples, len(ds))):
        example = ds[i]
        audio = np.asarray(example["audio"]["array"], dtype=np.float32)
        sr = int(example["audio"]["sampling_rate"])
        true_label = example["target_emotion"]

        t0 = time.perf_counter()
        out = pipe.predict_from_audio(audio, sr=sr)
        wall = time.perf_counter() - t0
        latencies.append(wall)

        print("\n" + "-" * 60)
        print(f"  Sample {i+1}  (MELD test — true: {true_label})")
        print(f"  ASR         : {_safe(out['text'])}")
        print(f"  Predicted   : {out['predicted_emotion']:<12s} "
              f"(conf={out['confidence']:.3f})")
        print(f"  Handover    : {out['should_handover']} "
              f"(score={out['handover_score']:.3f} / "
              f"threshold={out['threshold']:.3f})")
        print(f"  Total time  : {wall*1000:6.0f} ms  "
              f"(internal sum={out['timings']['total']*1000:.0f} ms)")
        stages = ("asr", "vader", "goemo", "roberta", "speechbrain", "meta")
        stage_str = "  ".join(
            f"{s}={out['timings'][s]*1000:.0f}ms" for s in stages
        )
        print(f"  Breakdown   : {stage_str}")

    print("\n" + "=" * 60)
    print("  Latency Summary")
    print("=" * 60)
    arr = np.array(latencies) * 1000.0
    print(f"  n={len(arr)}  mean={arr.mean():.0f} ms  "
          f"median={np.median(arr):.0f} ms  "
          f"p90={np.percentile(arr, 90):.0f} ms  "
          f"max={arr.max():.0f} ms")
    target_ms = 2000
    ok = (arr < target_ms).mean() * 100
    print(f"  {ok:.0f}% of utterances under the {target_ms} ms target "
          f"({'PASS' if arr.mean() < target_ms else 'FAIL — optimise'}).")


def main() -> None:
    print("=" * 60)
    print("  SmartHandover — Day 9: End-to-End Pipeline Smoke Test")
    print("=" * 60)
    pipe = SmartHandoverPipeline()

    # Run a few MELD samples with the full audio->handover chain
    _benchmark(pipe, n_samples=5)


if __name__ == "__main__":
    main()
