"""
SmartHandover — Day 4-5: RoBERTa Text-Only Fine-Tuning

TextOnlyClassifier: TextEncoder(roberta-base) -> [768] -> Linear(768,256) -> ReLU -> Dropout(0.3) -> Linear(256,5)

Training features:
  - WeightedRandomSampler for class imbalance
  - AdamW (lr=2e-5, weight_decay=0.01)
  - Linear warmup scheduler (10% steps)
  - FP16 autocast
  - Early stopping on weighted_f1 (patience=8)
  - Encoder freezing: frozen for first 2 epochs, then unfreeze top 4 layers
"""

import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from sklearn.metrics import f1_score
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.text_encoder import TextEncoder
from src.data.load_meld import TARGET_LABELS, TARGET_LABEL2ID

NUM_CLASSES = len(TARGET_LABELS)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TextOnlyClassifier(nn.Module):
    """RoBERTa text encoder + classification head."""

    def __init__(self, model_name="roberta-base", num_classes=NUM_CLASSES):
        super().__init__()
        self.encoder = TextEncoder(model_name=model_name)
        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        emb = self.encoder(input_ids, attention_mask)  # [B, 768]
        return self.head(emb)                           # [B, num_classes]

    def freeze_encoder(self):
        self.encoder.freeze()

    def unfreeze_top_layers(self, n=4):
        self.encoder.unfreeze_top_n(n)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MeldTextDataset(Dataset):
    """Simple text + label dataset from MELD."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def build_sampler(labels):
    """Build WeightedRandomSampler to handle class imbalance."""
    counts = Counter(labels)
    total = len(labels)
    class_weights = {cls: total / count for cls, count in counts.items()}
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_meld_texts(split: str):
    """Load MELD split and return (texts, labels) as lists."""
    from src.data.load_meld import load_meld
    ds = load_meld(split=split, streaming=False)
    texts, labels = [], []
    for ex in ds:
        texts.append(ex["text"])
        labels.append(TARGET_LABEL2ID[ex["target_emotion"]])
    return texts, labels


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    batch_size=16,
    max_epochs=40,
    lr=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    patience=8,
    freeze_epochs=2,
    unfreeze_top_n=4,
    checkpoint_dir="checkpoints",
    device=None,
):
    """Full training pipeline. Returns (model, history)."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Data ---
    print("Loading MELD splits...")
    train_texts, train_labels = load_meld_texts("train")
    val_texts, val_labels = load_meld_texts("validation")
    print(f"  Train: {len(train_texts)}, Val: {len(val_texts)}")

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    train_ds = MeldTextDataset(train_texts, train_labels, tokenizer)
    val_ds = MeldTextDataset(val_texts, val_labels, tokenizer)

    sampler = build_sampler(train_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                            num_workers=0, pin_memory=True)

    # --- Model ---
    print("Building TextOnlyClassifier...")
    model = TextOnlyClassifier()
    model.freeze_encoder()  # start frozen
    model.to(device)

    # --- Class weights for loss ---
    counts = Counter(train_labels)
    total = len(train_labels)
    class_weights = torch.tensor(
        [total / (NUM_CLASSES * counts[i]) for i in range(NUM_CLASSES)],
        dtype=torch.float32,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --- Optimizer (only trainable params initially = head + attn_pool) ---
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )

    total_steps = len(train_loader) * max_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # --- Training ---
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_f1 = 0.0
    patience_counter = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_wf1": [], "val_mf1": []}

    print(f"\nTraining for up to {max_epochs} epochs (patience={patience})...")
    print(f"  Encoder FROZEN for first {freeze_epochs} epochs, then unfreeze top {unfreeze_top_n} layers.\n")

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        # --- Unfreeze after freeze_epochs ---
        if epoch == freeze_epochs + 1:
            print(f"  >>> Unfreezing top {unfreeze_top_n} RoBERTa layers <<<")
            model.unfreeze_top_layers(unfreeze_top_n)
            # Rebuild optimizer with all trainable params
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr, weight_decay=weight_decay,
            )
            remaining_steps = len(train_loader) * (max_epochs - epoch + 1)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, int(remaining_steps * warmup_ratio), remaining_steps
            )

        # --- Train ---
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs} [train]", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{max_epochs} [val]", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)

                val_loss_sum += loss.item()
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss_sum / len(val_loader)
        wf1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        mf1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        elapsed = time.time() - t0
        frozen_str = "FROZEN" if epoch <= freeze_epochs else "UNFROZEN"
        print(f"  Epoch {epoch:>2d} | loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | "
              f"W-F1={wf1:.4f} | M-F1={mf1:.4f} | {elapsed:.0f}s | {frozen_str}")

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_wf1"].append(wf1)
        history["val_mf1"].append(mf1)

        # --- Early stopping ---
        if wf1 > best_f1:
            best_f1 = wf1
            patience_counter = 0
            ckpt_path = os.path.join(checkpoint_dir, "roberta_text_only.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"    -> New best W-F1={best_f1:.4f}, saved to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    print(f"\nTraining complete. Best val W-F1: {best_f1:.4f}")
    return model, history


# ---------------------------------------------------------------------------
# Evaluation on test set
# ---------------------------------------------------------------------------

def evaluate_on_test(model, device=None, batch_size=32, split="test"):
    """Evaluate model on a MELD split. Returns DataFrame with predictions."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    test_texts, test_labels = load_meld_texts(split)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    test_ds = MeldTextDataset(test_texts, test_labels, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating test set"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(input_ids, attention_mask)

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch["label"].numpy())

    # Build results DataFrame
    rows = []
    for i in range(len(test_texts)):
        row = {
            "text": test_texts[i],
            "true_label": TARGET_LABELS[all_labels[i]],
            "predicted_class": TARGET_LABELS[all_preds[i]],
        }
        for j, label in enumerate(TARGET_LABELS):
            row[f"prob_{label[:5]}"] = float(all_probs[i][j])
        rows.append(row)

    return pd.DataFrame(rows)
