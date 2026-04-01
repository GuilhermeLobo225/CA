"""
Smart Handover — Text Encoder
Wraps roberta-base with learned attention pooling.
Input: tokenised text (input_ids + attention_mask)
Output: [batch, 768] embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, AutoTokenizer


class TextEncoder(nn.Module):
    def __init__(self, model_name="roberta-base", embedding_dim=768):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embedding_dim = embedding_dim

        # Learned attention pooling: project each token to a scalar weight
        self.attn_pool = nn.Linear(embedding_dim, 1)

    def get_tokenizer(self):
        """Return tokenizer for use in DataLoader collate_fn."""
        return self.tokenizer

    def freeze(self):
        """Freeze all RoBERTa parameters (pooling layer stays trainable)."""
        for param in self.roberta.parameters():
            param.requires_grad = False

    def unfreeze_top_n(self, n):
        """Unfreeze the last n transformer layers of RoBERTa.

        RoBERTa-base has 12 layers: self.roberta.encoder.layer[0..11].
        Calling unfreeze_top_n(2) unfreezes layers 10 and 11.
        """
        total_layers = len(self.roberta.encoder.layer)
        for i in range(total_layers - n, total_layers):
            for param in self.roberta.encoder.layer[i].parameters():
                param.requires_grad = True

    def attention_pool(self, hidden_states, attention_mask):
        """Attention-weighted mean pooling.

        Args:
            hidden_states: [batch, seq_len, 768] from RoBERTa
            attention_mask: [batch, seq_len] — 1 for real tokens, 0 for padding

        Returns:
            pooled: [batch, 768]
        """
        # Compute per-token attention scores
        attn_scores = self.attn_pool(hidden_states).squeeze(-1)   # [batch, seq_len]

        # Mask out padding positions with -inf before softmax
        attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))

        # Normalise to get weights
        attn_weights = F.softmax(attn_scores, dim=-1)             # [batch, seq_len]

        # Weighted sum
        pooled = torch.bmm(
            attn_weights.unsqueeze(1),   # [batch, 1, seq_len]
            hidden_states,               # [batch, seq_len, 768]
        ).squeeze(1)                     # [batch, 768]

        return pooled

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids:      [batch, seq_len]  — tokenised text
            attention_mask: [batch, seq_len]  — 1 for real, 0 for pad

        Returns:
            [batch, 768] pooled text embedding
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, 768]
        pooled = self.attention_pool(hidden_states, attention_mask)
        return pooled
