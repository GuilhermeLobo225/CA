"""SmartHandover — v5 text training with focal loss.

Near-copy of ``src/training/train_text.py`` with two differences:

  * ``FocalLoss(alpha=class_weights, gamma=2.0)`` replaces
    ``nn.CrossEntropyLoss(weight=class_weights)``.
  * Early stopping monitors **val frustration recall** rather than
    weighted F1 — we are explicitly trading some macro performance for
    frust recall.

All other hyperparameters (lr, warmup, freeze schedule, AdamW,
gradient clipping, fp16, weighted sampler) are unchanged. The script
re-uses the helpers in ``train_text`` for data loading so that the v5
and v4 conditions only differ in loss / early-stopping objective.
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.load_meld import TARGET_LABEL2ID, TARGET_LABELS  # noqa: E402
from src.training.focal_loss import FocalLoss  # noqa: E402
from src.training.train_text import (  # noqa: E402
    MeldTextDataset,
    NUM_CLASSES,
    TextOnlyClassifier,
    build_sampler,
)

FRUST_ID = TARGET_LABEL2ID["frustration"]


def train_model_v5(
    train_data: Tuple,
    val_data: Tuple,
    batch_size: int = 16,
    max_epochs: int = 10,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    patience: int = 5,
    freeze_epochs: int = 2,
    unfreeze_top_n: int = 4,
    gamma: float = 2.0,
    checkpoint_dir: str = "checkpoints",
    checkpoint_name: str = "roberta_text_v5.pt",
    device: str | None = None,
    use_class_weights: bool = True,
) -> Tuple[nn.Module, Dict]:
    """Train RoBERTa with focal loss; early-stop on val frust recall."""

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_texts, train_labels = train_data
    val_texts, val_labels = val_data
    print(f"  Train={len(train_texts)}, Val={len(val_texts)}")

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    train_ds = MeldTextDataset(train_texts, train_labels, tokenizer)
    val_ds = MeldTextDataset(val_texts, val_labels, tokenizer)
    sampler = build_sampler(train_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                                sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2,
                              shuffle=False, num_workers=0, pin_memory=True)

    print("Building TextOnlyClassifier ...")
    model = TextOnlyClassifier()
    model.freeze_encoder()
    model.to(device)

    if use_class_weights:
        counts = Counter(train_labels)
        total = len(train_labels)
        class_weights = torch.tensor(
            [total / (NUM_CLASSES * counts.get(i, 1)) for i in range(NUM_CLASSES)],
            dtype=torch.float32,
        ).to(device)
    else:
        class_weights = None
    criterion = FocalLoss(alpha=class_weights, gamma=gamma)
    print(f"  Loss: FocalLoss(gamma={gamma}, alpha={class_weights})")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    total_steps = len(train_loader) * max_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_frust = -1.0
    best_state = None
    best_epoch = -1
    patience_counter = 0
    history = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "val_wf1": [], "val_mf1": [], "val_frust_recall": [],
    }

    print(f"\nTraining up to {max_epochs} epochs (patience={patience}, monitor=val_frust_recall)")
    print(f"  Encoder FROZEN for first {freeze_epochs} epochs, then unfreeze top {unfreeze_top_n}.\n")

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        if epoch == freeze_epochs + 1:
            print(f"  >>> Unfreezing top {unfreeze_top_n} RoBERTa layers <<<")
            model.unfreeze_top_layers(unfreeze_top_n)
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr, weight_decay=weight_decay,
            )
            remaining_steps = len(train_loader) * (max_epochs - epoch + 1)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, int(remaining_steps * warmup_ratio), remaining_steps,
            )

        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader,
                            desc=f"Epoch {epoch}/{max_epochs} [train]", leave=False):
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

        model.eval()
        val_loss_sum = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader,
                                desc=f"Epoch {epoch}/{max_epochs} [val]", leave=False):
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
        frust_recall = recall_score(
            all_labels, all_preds, labels=[FRUST_ID], average="macro", zero_division=0,
        )

        frozen_str = "FROZEN" if epoch <= freeze_epochs else "UNFROZEN"
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:>2d} | tr_loss={avg_train_loss:.4f} "
              f"| val_loss={avg_val_loss:.4f} | W-F1={wf1:.4f} | M-F1={mf1:.4f} "
              f"| FrustR={frust_recall:.4f} | {elapsed:.0f}s | {frozen_str}")

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_wf1"].append(wf1)
        history["val_mf1"].append(mf1)
        history["val_frust_recall"].append(frust_recall)

        if frust_recall > best_frust:
            best_frust = frust_recall
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
            ckpt_path = os.path.join(checkpoint_dir, checkpoint_name)
            torch.save(model.state_dict(), ckpt_path)
            print(f"    -> New best FrustR={best_frust:.4f}, saved to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (no FrustR improvement for {patience} epochs).")
                break

    print(f"\nTraining complete. Best val FrustR={best_frust:.4f} at epoch {best_epoch}.")
    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_val_frust_recall"] = best_frust
    return model, history
