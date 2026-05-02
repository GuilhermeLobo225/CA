"""
Smart Handover — Training Loop
Full training pipeline with FP16 autocast, gradient accumulation,
weighted loss, early stopping, and checkpointing.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
from functools import partial
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.load_meld import load_meld, TARGET_LABELS
from src.models.fusion_model import MultimodalFusionModel, FocalLoss
from src.evaluation.metrics import compute_class_weights, compute_metrics, print_metrics


# ──────────────────────────────────────────────────────────
#  COLLATE FUNCTION
# ──────────────────────────────────────────────────────────

def collate_fn(batch, tokenizer, audio_processor, max_text_length, max_audio_samples):
    """Custom collate for the multimodal DataLoader.

    Args:
        batch: list of HuggingFace dataset examples, each with
               'text', 'audio' (dict with 'array' + 'sampling_rate'), 'target_label'.
        tokenizer: RoBERTa tokenizer.
        audio_processor: Wav2Vec2Processor.
        max_text_length: max tokens for text.
        max_audio_samples: max raw waveform samples for audio.

    Returns:
        dict with tensors ready for the model.
    """
    texts = [item["text"] for item in batch]
    audios = [item["audio"]["array"] for item in batch]
    labels = [item["target_label"] for item in batch]

    # Truncate audio to max length
    audios = [a[:max_audio_samples] if len(a) > max_audio_samples else a for a in audios]

    # Tokenise text
    text_enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_text_length,
        return_tensors="pt",
    )

    # Process audio (padding + normalisation)
    audio_enc = audio_processor(
        audios,
        sampling_rate=16_000,
        padding=True,
        return_tensors="pt",
    )

    return {
        "text_input_ids": text_enc["input_ids"],
        "text_attention_mask": text_enc["attention_mask"],
        "audio_input_values": audio_enc["input_values"],
        "audio_attention_mask": audio_enc.get("attention_mask", None),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# ──────────────────────────────────────────────────────────
#  DATA LOADERS
# ──────────────────────────────────────────────────────────

def create_dataloaders(config, tokenizer, audio_processor):
    """Create train and validation DataLoaders."""
    max_text_length = config["data"]["max_text_length"]
    max_audio_samples = config["data"]["max_audio_length_sec"] * config["data"]["audio_sample_rate"]
    batch_size = config["training"]["batch_size"]

    train_ds = load_meld("train", streaming=False)
    val_ds = load_meld("validation", streaming=False)

    _collate = partial(
        collate_fn,
        tokenizer=tokenizer,
        audio_processor=audio_processor,
        max_text_length=max_text_length,
        max_audio_samples=max_audio_samples,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=0,       # HF Dataset + audio decoding = safer with 0
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader, train_ds


# ──────────────────────────────────────────────────────────
#  OPTIMIZER BUILDER
# ──────────────────────────────────────────────────────────

def build_optimizer(model, config):
    """Build AdamW optimizer with separate parameter groups.

    - Encoder params: lr * encoder_lr_multiplier
    - Fusion / classification head params: lr
    """
    lr = config["training"]["learning_rate"]
    encoder_lr = lr * config["training"]["encoder_lr_multiplier"]
    weight_decay = config["training"]["weight_decay"]

    # Separate encoder params from head params
    encoder_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "text_encoder.roberta" in name or "audio_encoder.wav2vec2" in name:
            encoder_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {"params": head_params, "lr": lr},
        {"params": encoder_params, "lr": encoder_lr},
    ]

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


# ──────────────────────────────────────────────────────────
#  VALIDATION
# ──────────────────────────────────────────────────────────

def validate(model, val_loader, criterion, device):
    """Run validation and return metrics dict."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        for batch in val_loader:
            text_ids = batch["text_input_ids"].to(device)
            text_mask = batch["text_attention_mask"].to(device)
            audio_vals = batch["audio_input_values"].to(device)
            audio_mask = batch["audio_attention_mask"]
            if audio_mask is not None:
                audio_mask = audio_mask.to(device)
            labels = batch["labels"].to(device)

            logits = model(text_ids, text_mask, audio_vals, audio_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            num_batches += 1

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["val_loss"] = total_loss / max(num_batches, 1)
    return metrics


# ──────────────────────────────────────────────────────────
#  CHECKPOINT UTILITIES
# ──────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, metrics, config):
    """Save model checkpoint."""
    ckpt_dir = config["training"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, "best_model.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": {k: v for k, v in metrics.items() if k != "confusion_matrix"},
        },
        path,
    )
    print(f"  >> Checkpoint saved to {path}")


# ──────────────────────────────────────────────────────────
#  MAIN TRAINING LOOP
# ──────────────────────────────────────────────────────────

def train(config):
    """Full training pipeline."""

    # ── Seed ──
    seed = config["hardware"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(config["hardware"]["device"])

    # ── Model ──
    print("Initialising model...")
    model = MultimodalFusionModel(config)
    model.to(device)

    # Grab tokenizer and processor before potentially freezing
    tokenizer = model.text_encoder.get_tokenizer()
    audio_processor = model.audio_encoder.get_processor()

    # ── Freeze encoders if configured ──
    if config["training"]["freeze_encoders"]:
        print(f"Freezing encoders for first {config['training']['freeze_epochs']} epochs.")
        model.freeze_encoders()

    # ── Data ──
    print("Loading data...")
    train_loader, val_loader, train_ds = create_dataloaders(config, tokenizer, audio_processor)

    # ── Class weights ──
    class_weights = compute_class_weights(train_ds).to(device)
    print(f"Class weights: {dict(zip(TARGET_LABELS, class_weights.tolist()))}")

    # ── Loss ──
    if config["training"]["use_focal_loss"]:
        criterion = FocalLoss(
            weight=class_weights if config["training"]["use_weighted_loss"] else None,
            gamma=config["training"]["focal_gamma"],
        )
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights if config["training"]["use_weighted_loss"] else None,
        )

    # ── Optimizer & Scheduler ──
    optimizer = build_optimizer(model, config)

    accum_steps = config["training"]["accumulation_steps"]
    total_steps = (len(train_loader) // accum_steps) * config["training"]["epochs"]
    warmup_steps = int(config["training"]["warmup_ratio"] * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── GradScaler for FP16 ──
    scaler = torch.amp.GradScaler()

    # ── Training ──
    best_metric = -1.0  # start at -1 so first epoch always saves
    patience_counter = 0
    patience = config["training"]["early_stopping_patience"]
    max_grad_norm = config["training"]["max_grad_norm"]
    freeze_epochs = config["training"]["freeze_epochs"]

    print(f"\nStarting training for {config['training']['epochs']} epochs...")
    print(f"Effective batch size: {config['training']['batch_size']} x {accum_steps} = "
          f"{config['training']['batch_size'] * accum_steps}")

    for epoch in range(config["training"]["epochs"]):
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch + 1}/{config['training']['epochs']}")
        print(f"{'='*60}")

        # ── Unfreeze encoders after freeze_epochs ──
        if (
            config["training"]["freeze_encoders"]
            and epoch == freeze_epochs
        ):
            n = config["training"]["unfreeze_top_n_layers"]
            print(f"  >> Unfreezing top {n} layers of both encoders.")
            model.unfreeze_encoders(n)
            # Rebuild optimizer to include newly unfrozen params
            optimizer = build_optimizer(model, config)
            # Recompute scheduler for remaining epochs
            remaining_steps = (len(train_loader) // accum_steps) * (config["training"]["epochs"] - epoch)
            warmup_remaining = int(config["training"]["warmup_ratio"] * remaining_steps)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_remaining,
                num_training_steps=remaining_steps,
            )

        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0

        progress = tqdm(enumerate(train_loader), total=len(train_loader), desc="  Training")
        for step, batch in progress:
            text_ids = batch["text_input_ids"].to(device)
            text_mask = batch["text_attention_mask"].to(device)
            audio_vals = batch["audio_input_values"].to(device)
            audio_mask = batch["audio_attention_mask"]
            if audio_mask is not None:
                audio_mask = audio_mask.to(device)
            labels = batch["labels"].to(device)

            # Forward pass with FP16
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(text_ids, text_mask, audio_vals, audio_mask)
                loss = criterion(logits, labels)
                loss = loss / accum_steps  # normalise for accumulation

            # Backward
            scaler.scale(loss).backward()

            epoch_loss += loss.item() * accum_steps  # undo normalisation for logging

            # Optimiser step at accumulation boundary
            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            progress.set_postfix({"loss": f"{loss.item() * accum_steps:.4f}"})

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"  Train loss: {avg_train_loss:.4f}")

        # ── Validation ──
        print("  Validating...")
        val_metrics = validate(model, val_loader, criterion, device)
        print_metrics(val_metrics)

        # ── Early stopping ──
        current_metric = val_metrics["frustration_recall"]
        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            if config["training"]["save_best_only"]:
                save_checkpoint(model, optimizer, epoch, val_metrics, config)
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("  >> Early stopping triggered.")
                break

    print(f"\nTraining complete. Best frustration recall: {best_metric:.4f}")


# ──────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    train(config)
