"""SmartHandover - Wav2Vec2 fine-tuning module (Phase 7).

Replaces the frozen ``superb/wav2vec2-large-superb-er`` (which is
saturated for ang and blind to sad in synthetic distribution) with
a checkpoint fine-tuned on a multi-corpus mix:

  * MELD audio       (real, sitcom origin)
  * CREMA-D audio    (real, 91 actors, strong sadness signal)
  * synthetic audio_phone (telephone-band degraded TTS, optional)

Training strategy mirrors the RoBERTa text fine-tune:
  * encoder frozen for ``freeze_epochs`` epochs (head learns first)
  * then unfreeze the top ``unfreeze_top_n`` transformer layers
  * AdamW + linear warmup, FP16 autocast, early stopping on MELD val W-F1

This module exposes building blocks; the entry point is
``scripts/run_dayF13_finetune_wav2vec2.py``.
"""

from __future__ import annotations

import os
import time
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

from sklearn.metrics import f1_score

from src.data.load_meld import TARGET_LABELS, TARGET_LABEL2ID
from src.evaluation.metrics import compute_metrics


NUM_CLASSES = len(TARGET_LABELS)
MODEL_ID = "superb/wav2vec2-large-superb-er"
TARGET_SR = 16000
MAX_AUDIO_SEC = 8.0   # cap clips at 8 s to keep VRAM predictable


# ===========================================================================
# Dataset
# ===========================================================================


class AudioRecord:
    """Lightweight record describing one training clip.

    Attributes:
        path:        absolute path to a .wav file (lazy-loaded)
        label_id:    int 0..4
        source:      'meld' | 'cremad' | 'synth' (for sample weighting)
        weight:      float multiplier for sampler probability
    """
    __slots__ = ("path", "label_id", "source", "weight")

    def __init__(self, path: str, label_id: int, source: str,
                 weight: float = 1.0):
        self.path = path
        self.label_id = label_id
        self.source = source
        self.weight = weight


class AudioDataset(Dataset):
    """Loads waveforms on-the-fly, crops/pads to MAX_AUDIO_SEC, normalises.

    Works for any source whose records share the same target label space.
    Audio normalisation: zero-mean unit-RMS, then clip to [-1, 1].
    """

    def __init__(self, records: List[AudioRecord], feature_extractor,
                 max_seconds: float = MAX_AUDIO_SEC, sr: int = TARGET_SR):
        self.records = records
        self.fe = feature_extractor
        self.max_samples = int(round(max_seconds * sr))
        self.sr = sr

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        wav = _read_wav(rec.path, target_sr=self.sr)
        wav = _crop_or_pad(wav, self.max_samples)

        # Wav2Vec2 feature extractor wants raw float32 in [-1, 1].
        inputs = self.fe(
            wav, sampling_rate=self.sr, return_tensors="pt",
            padding=False,
        )
        # input_values: [1, T] -> [T]
        input_values = inputs["input_values"].squeeze(0).to(torch.float32)
        attn = inputs.get("attention_mask")
        if attn is not None:
            attn = attn.squeeze(0).to(torch.long)
        else:
            attn = torch.ones_like(input_values, dtype=torch.long)

        return {
            "input_values":   input_values,
            "attention_mask": attn,
            "label":          torch.tensor(rec.label_id, dtype=torch.long),
        }


def _read_wav(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        try:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(sr, target_sr)
            wav = resample_poly(wav, target_sr // g, sr // g)
        except ImportError:
            pass  # fall through with original sr; rare path
    # zero-mean unit-RMS
    wav = wav - np.mean(wav)
    rms = float(np.sqrt(np.mean(wav ** 2) + 1e-12))
    if rms > 0:
        wav = wav / rms * 0.1   # 0.1 RMS = comfortable in [-1, 1]
    return wav.astype(np.float32)


def _crop_or_pad(wav: np.ndarray, max_samples: int) -> np.ndarray:
    if len(wav) > max_samples:
        # Centre-crop: keep the middle of the clip (more emotionally salient)
        start = (len(wav) - max_samples) // 2
        return wav[start:start + max_samples]
    if len(wav) < max_samples:
        return np.pad(wav, (0, max_samples - len(wav)))
    return wav


def collate_pad(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Stack equal-length tensors (we already pad in _crop_or_pad)."""
    return {
        "input_values":   torch.stack([b["input_values"]   for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "label":          torch.stack([b["label"]          for b in batch]),
    }


# ===========================================================================
# Per-source loaders -> List[AudioRecord]
# ===========================================================================


def load_meld_audio_records(splits: Tuple[str, ...] = ("train",),
                             cache_dir: Optional[str] = None
                             ) -> List[AudioRecord]:
    """Pull MELD audio records straight from the HuggingFace dataset.

    The HF MELD audio dataset stores clips as ``audio = {array, sampling_rate}``
    rather than as files. We materialise them on disk under
    ``cache_dir`` (default: data/cache/meld_audio/) the first time
    they're needed and reuse the cached .wav afterwards.
    """
    from src.data.load_meld import load_meld
    cache_dir = cache_dir or os.path.join("data", "cache", "meld_audio")
    os.makedirs(cache_dir, exist_ok=True)

    records: List[AudioRecord] = []
    for split in splits:
        ds = load_meld(split=split, streaming=False)
        split_dir = os.path.join(cache_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for i, ex in enumerate(ds):
            target = ex.get("target_emotion")
            if target not in TARGET_LABEL2ID:
                continue
            label_id = TARGET_LABEL2ID[target]
            path = os.path.join(split_dir, f"{split}_{i:06d}.wav")
            if not os.path.exists(path):
                _write_wav(path, ex["audio"]["array"],
                            int(ex["audio"]["sampling_rate"]))
            records.append(AudioRecord(path=path, label_id=label_id,
                                         source="meld"))
    return records


def load_cremad_audio_records(split: str = "train") -> List[AudioRecord]:
    """Load CREMA-D records for the requested speaker-disjoint split."""
    from src.data.load_cremad import CremadLoader, speaker_disjoint_split
    loader = CremadLoader()
    splits = speaker_disjoint_split(loader)
    if split not in splits:
        raise KeyError(f"Unknown CREMA-D split '{split}'. "
                       f"Available: {list(splits)}")
    indices = splits[split]

    audio_root = loader.audio_dir
    records: List[AudioRecord] = []
    for idx in indices:
        rec_meta = loader._records[idx]   # internal lookup, no audio decode
        target_label = loader.label_map.get(rec_meta["emotion_code"])
        if target_label is None:
            continue
        if target_label not in TARGET_LABEL2ID:
            continue
        full_path = os.path.join(audio_root, rec_meta["filename"])
        records.append(AudioRecord(
            path=full_path,
            label_id=TARGET_LABEL2ID[target_label],
            source="cremad",
        ))
    return records


def load_synth_phone_records(audio_root: str = None,
                              weight: float = 0.5
                              ) -> List[AudioRecord]:
    """Load the phone-band-degraded synthetic clips.

    ``weight=0.5`` halves their probability in the WeightedRandomSampler,
    so a mostly-real batch is preserved while the network still sees
    the synthetic distribution. Bump to 1.0 to weight equally with real.
    """
    audio_root = audio_root or os.path.join("data", "synthetic", "audio_phone")
    if not os.path.isdir(audio_root):
        return []
    records: List[AudioRecord] = []
    for label in os.listdir(audio_root):
        sub = os.path.join(audio_root, label)
        if not os.path.isdir(sub) or label not in TARGET_LABEL2ID:
            continue
        label_id = TARGET_LABEL2ID[label]
        for fname in os.listdir(sub):
            if not fname.lower().endswith(".wav"):
                continue
            records.append(AudioRecord(
                path=os.path.join(sub, fname),
                label_id=label_id, source="synth", weight=weight,
            ))
    return records


def _write_wav(path: str, audio: np.ndarray, sr: int) -> None:
    import soundfile as sf
    sf.write(path, audio.astype(np.float32), sr, subtype="PCM_16")


# ===========================================================================
# Sampler / criterion helpers
# ===========================================================================


def build_weighted_sampler(records: List[AudioRecord]) -> WeightedRandomSampler:
    """Combine inverse-frequency class weighting with per-source weight.

    Final per-sample weight = source_weight * (1 / class_count).
    """
    counts = Counter(r.label_id for r in records)
    total = sum(counts.values())
    sample_w = []
    for r in records:
        cls_w = total / max(counts[r.label_id], 1)
        sample_w.append(r.weight * cls_w)
    return WeightedRandomSampler(weights=sample_w,
                                  num_samples=len(records),
                                  replacement=True)


def class_weights(records: List[AudioRecord]) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropy."""
    counts = Counter(r.label_id for r in records)
    total = sum(counts.values())
    return torch.tensor(
        [total / (NUM_CLASSES * counts.get(i, 1)) for i in range(NUM_CLASSES)],
        dtype=torch.float32,
    )


# ===========================================================================
# Model: HF Wav2Vec2 with a 5-class head
# ===========================================================================


def build_model(num_classes: int = NUM_CLASSES,
                model_id: str = MODEL_ID,
                device: Optional[str] = None):
    """Load wav2vec2-large-superb-er with a fresh 5-class head.

    The IEMOCAP-trained head (4 classes) is dropped because our label
    space is different (frustration is novel). The encoder is kept;
    that's where the emotion prosody features live.
    """
    from transformers import (
        Wav2Vec2ForSequenceClassification,
        Wav2Vec2FeatureExtractor,
    )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,   # drop the 4-class IEMOCAP head
    )
    model.to(device)
    return model, feature_extractor


def freeze_encoder(model) -> None:
    """Freeze the wav2vec2 encoder; head + projector remain trainable."""
    for p in model.wav2vec2.parameters():
        p.requires_grad = False


def unfreeze_top_layers(model, n: int) -> None:
    """Unfreeze the top ``n`` transformer layers of the encoder."""
    layers = model.wav2vec2.encoder.layers
    n = max(0, min(n, len(layers)))
    for layer in layers[-n:]:
        for p in layer.parameters():
            p.requires_grad = True


# ===========================================================================
# Train / eval loops
# ===========================================================================


def evaluate(model, loader, device, criterion=None) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_values   = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                out = model(input_values=input_values,
                             attention_mask=attention_mask,
                             labels=labels if criterion is None else None)
                if criterion is None:
                    loss = out.loss
                else:
                    loss = criterion(out.logits, labels)
            total_loss += float(loss.item())
            n_batches += 1
            all_preds.extend(out.logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    if not all_labels:
        return {"loss": 0.0, "weighted_f1": 0.0, "macro_f1": 0.0,
                "frust_recall": 0.0, "n": 0}
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds),
                                target_names=TARGET_LABELS)
    return {
        "loss":          total_loss / max(n_batches, 1),
        "weighted_f1":   metrics["weighted_f1"],
        "macro_f1":      metrics["macro_f1"],
        "frust_recall":  metrics["frustration_recall"],
        "n":             len(all_labels),
    }


def train_audio_model(
    train_records: List[AudioRecord],
    val_records:   List[AudioRecord],
    *,
    checkpoint_path: str,
    max_epochs: int = 8,
    batch_size: int = 4,
    accumulation_steps: int = 4,
    lr_head: float = 1e-4,
    lr_encoder: float = 5e-6,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    freeze_epochs: int = 2,
    unfreeze_top_n: int = 4,
    patience: int = 4,
    use_class_weights: bool = True,
    device: Optional[str] = None,
    log_every: int = 50,
) -> Dict:
    """Full fine-tune loop. Returns a history dict + best metrics."""
    from transformers import get_linear_schedule_with_warmup

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")
    print(f"  train: {len(train_records)} | val: {len(val_records)}")

    model, feature_extractor = build_model(device=device)
    freeze_encoder(model)

    train_ds = AudioDataset(train_records, feature_extractor)
    val_ds   = AudioDataset(val_records,   feature_extractor)

    sampler = build_weighted_sampler(train_records)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, collate_fn=collate_pad,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size * 2,
                              shuffle=False, collate_fn=collate_pad,
                              num_workers=0, pin_memory=True)

    if use_class_weights:
        cw = class_weights(train_records).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw)
    else:
        criterion = nn.CrossEntropyLoss()

    # Two parameter groups: head (high lr), encoder (low lr).
    head_params = [p for n, p in model.named_parameters()
                    if not n.startswith("wav2vec2.")]
    enc_params  = [p for n, p in model.named_parameters()
                    if n.startswith("wav2vec2.")]
    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": lr_head},
         {"params": enc_params,  "lr": lr_encoder}],
        weight_decay=weight_decay,
    )

    # Effective steps account for gradient accumulation.
    eff_steps_per_epoch = max(len(train_loader) // accumulation_steps, 1)
    total_steps = eff_steps_per_epoch * max_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps,
                                                  total_steps)

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    history = {
        "epoch":         [], "train_loss": [],
        "val_loss":      [], "val_wf1":    [],
        "val_mf1":       [], "val_frust_r":[],
    }

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    best_wf1 = -1.0
    patience_counter = 0

    print(f"  max_epochs={max_epochs}  freeze_epochs={freeze_epochs}  "
          f"patience={patience}")

    for epoch in range(1, max_epochs + 1):
        if epoch == freeze_epochs + 1:
            print(f"  >>> Unfreezing top {unfreeze_top_n} layers <<<")
            unfreeze_top_layers(model, unfreeze_top_n)
            # Re-create optimizer to include the freshly-trainable params
            head_params = [p for n, p in model.named_parameters()
                            if not n.startswith("wav2vec2.")
                            and p.requires_grad]
            enc_params  = [p for n, p in model.named_parameters()
                            if n.startswith("wav2vec2.") and p.requires_grad]
            optimizer = torch.optim.AdamW(
                [{"params": head_params, "lr": lr_head},
                 {"params": enc_params,  "lr": lr_encoder}],
                weight_decay=weight_decay,
            )
            remaining_steps = eff_steps_per_epoch * (max_epochs - epoch + 1)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, int(remaining_steps * warmup_ratio),
                remaining_steps,
            )

        model.train()
        running_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader,
                                           desc=f"ep{epoch}/{max_epochs} train",
                                           leave=False)):
            input_values   = batch["input_values"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels         = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                out = model(input_values=input_values,
                             attention_mask=attention_mask)
                loss = criterion(out.logits, labels) / accumulation_steps

            scaler.scale(loss).backward()
            running_loss += float(loss.item()) * accumulation_steps

            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0,
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = running_loss / max(len(train_loader), 1)

        val = evaluate(model, val_loader, device, criterion=criterion)
        elapsed = time.time() - t0
        frozen_str = "FROZEN" if epoch <= freeze_epochs else "UNFROZEN"
        print(f"  ep {epoch:>2d} | train_loss={avg_train_loss:.4f} | "
              f"val_loss={val['loss']:.4f} | "
              f"W-F1={val['weighted_f1']:.4f} | "
              f"M-F1={val['macro_f1']:.4f} | "
              f"FrustR={val['frust_recall']:.4f} | "
              f"{elapsed:.0f}s | {frozen_str}")

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val["loss"])
        history["val_wf1"].append(val["weighted_f1"])
        history["val_mf1"].append(val["macro_f1"])
        history["val_frust_r"].append(val["frust_recall"])

        if val["weighted_f1"] > best_wf1:
            best_wf1 = val["weighted_f1"]
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"    -> new best W-F1={best_wf1:.4f}, saved {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs).")
                break

    print(f"\n  Training complete. Best val W-F1 = {best_wf1:.4f}")
    return {"history": history, "best_val_wf1": best_wf1}


def evaluate_checkpoint(checkpoint_path: str,
                        records: List[AudioRecord],
                        batch_size: int = 8,
                        device: Optional[str] = None) -> Dict[str, float]:
    """Load a checkpoint, run on a record set, return metrics dict."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, fe = build_model(device=device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device,
                                       weights_only=True))
    ds = AudioDataset(records, fe)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                         collate_fn=collate_pad, num_workers=0,
                         pin_memory=True)
    return evaluate(model, loader, device)


def predict_records(checkpoint_path: str,
                    records: List[AudioRecord],
                    batch_size: int = 8,
                    device: Optional[str] = None) -> List[Dict]:
    """Predict probabilities on a record list. Returns list of dicts:
       {path, source, label_id, true_label, p_anger, p_frust, p_sad,
        p_neut, p_satis, predicted_class}.

    Used by Phase-10 (ensemble v3) to materialise wav2vec2 fine-tuned
    predictions over the MELD audio so they can replace the
    speechbrain_predictions.csv columns.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, fe = build_model(device=device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device,
                                       weights_only=True))
    ds = AudioDataset(records, fe)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                         collate_fn=collate_pad, num_workers=0,
                         pin_memory=True)

    out_rows: List[Dict] = []
    rec_iter = iter(records)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict"):
            input_values   = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                out = model(input_values=input_values,
                             attention_mask=attention_mask)
            probs = torch.softmax(out.logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)
            for i in range(len(probs)):
                rec = next(rec_iter)
                row = {
                    "path":            rec.path,
                    "source":          rec.source,
                    "true_label_id":   rec.label_id,
                    "true_label":      TARGET_LABELS[rec.label_id],
                    "predicted_class": TARGET_LABELS[int(preds[i])],
                    "p_anger":         float(probs[i][0]),
                    "p_frust":         float(probs[i][1]),
                    "p_sad":           float(probs[i][2]),
                    "p_neut":          float(probs[i][3]),
                    "p_satis":         float(probs[i][4]),
                }
                out_rows.append(row)
    return out_rows
