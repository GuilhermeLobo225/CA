"""
Smart Handover — Multimodal Fusion Model
Late fusion of text (RoBERTa) and audio (Wav2Vec2) embeddings.
Supports concat and gated fusion strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.text_encoder import TextEncoder
from src.models.audio_encoder import AudioEncoder


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    Reduces the loss contribution from easy/confident predictions,
    focusing training on hard minority samples (e.g., frustration).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight  # class weights tensor

    def forward(self, logits, targets):
        """
        Args:
            logits:  [batch, num_classes] raw logits
            targets: [batch] integer class labels
        """
        ce_loss = F.cross_entropy(
            logits, targets, weight=self.weight, reduction="none"
        )
        pt = torch.exp(-ce_loss)  # probability of the correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class MultimodalFusionModel(nn.Module):
    """Multimodal late fusion model for emotion classification.

    Architecture:
        text  -> TextEncoder  -> [batch, 768] --+
                                                 |--> fusion --> [batch, num_classes]
        audio -> AudioEncoder -> [batch, 768] --+

    Fusion types:
        - "concat": concatenate [text; audio] -> MLP -> logits
        - "gated":  learned gate weights text vs audio -> MLP -> logits
    """

    def __init__(self, config):
        super().__init__()

        # Encoders
        self.text_encoder = TextEncoder(
            model_name=config["text_encoder"]["model_name"],
            embedding_dim=config["text_encoder"]["embedding_dim"],
        )
        self.audio_encoder = AudioEncoder(
            model_name=config["audio_encoder"]["model_name"],
            embedding_dim=config["audio_encoder"]["embedding_dim"],
        )

        emb_dim = config["text_encoder"]["embedding_dim"]  # 768
        fusion_type = config["fusion"]["type"]
        hidden_dim = config["fusion"]["hidden_dim"]
        dropout = config["fusion"]["dropout"]
        num_classes = config["fusion"]["num_classes"]

        self.fusion_type = fusion_type

        if fusion_type == "concat":
            # [768 + 768] = 1536 -> hidden_dim -> num_classes
            self.classifier = nn.Sequential(
                nn.Linear(emb_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        elif fusion_type == "gated":
            # Gate learns per-dimension weighting between text and audio
            self.gate = nn.Linear(emb_dim * 2, emb_dim)
            self.classifier = nn.Sequential(
                nn.Linear(emb_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def freeze_encoders(self):
        """Freeze both encoder backbones."""
        self.text_encoder.freeze()
        self.audio_encoder.freeze()

    def unfreeze_encoders(self, n_layers):
        """Unfreeze top n transformer layers of both encoders."""
        self.text_encoder.unfreeze_top_n(n_layers)
        self.audio_encoder.unfreeze_top_n(n_layers)

    def forward(
        self,
        text_input_ids,
        text_attention_mask,
        audio_input_values,
        audio_attention_mask=None,
    ):
        """
        Args:
            text_input_ids:      [batch, seq_len]
            text_attention_mask:  [batch, seq_len]
            audio_input_values:   [batch, waveform_len]
            audio_attention_mask: [batch, waveform_len] or None

        Returns:
            logits: [batch, num_classes]
        """
        text_emb = self.text_encoder(text_input_ids, text_attention_mask)     # [batch, 768]
        audio_emb = self.audio_encoder(audio_input_values, audio_attention_mask)  # [batch, 768]

        if self.fusion_type == "concat":
            combined = torch.cat([text_emb, audio_emb], dim=-1)  # [batch, 1536]
            logits = self.classifier(combined)

        elif self.fusion_type == "gated":
            combined = torch.cat([text_emb, audio_emb], dim=-1)  # [batch, 1536]
            gate = torch.sigmoid(self.gate(combined))            # [batch, 768]
            fused = gate * text_emb + (1 - gate) * audio_emb    # [batch, 768]
            logits = self.classifier(fused)

        return logits
