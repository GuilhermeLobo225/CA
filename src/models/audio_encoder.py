"""
Smart Handover — Audio Encoder
Wraps facebook/wav2vec2-base with learned attention pooling.
Input: raw waveform tensors (16 kHz)
Output: [batch, 768] embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Processor


class AudioEncoder(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base", embedding_dim=768):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base",
            mask_time_prob=0.0,
            mask_feature_prob=0.0 
        )
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.embedding_dim = embedding_dim

        # Learned attention pooling
        self.attn_pool = nn.Linear(embedding_dim, 1)

    def get_processor(self):
        """Return processor for use in DataLoader collate_fn."""
        return self.processor

    def freeze(self):
        """Freeze all wav2vec2 parameters (pooling layer stays trainable).
        Also freezes the CNN feature extractor which should never be fine-tuned.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def unfreeze_top_n(self, n):
        """Unfreeze the last n transformer layers of wav2vec2.

        wav2vec2-base has 12 layers: self.wav2vec2.encoder.layers[0..11].
        Calling unfreeze_top_n(2) unfreezes layers 10 and 11.
        The CNN feature extractor stays frozen.
        """
        total_layers = len(self.wav2vec2.encoder.layers)
        for i in range(total_layers - n, total_layers):
            for param in self.wav2vec2.encoder.layers[i].parameters():
                param.requires_grad = True

    def _compute_frame_mask(self, attention_mask, hidden_states):
        """Derive a frame-level mask from the waveform-level attention mask.

        wav2vec2's CNN feature extractor downsamples the raw waveform.
        We need to reduce the attention mask to match the hidden_states length.

        Args:
            attention_mask: [batch, raw_waveform_len] or None
            hidden_states:  [batch, num_frames, 768]

        Returns:
            frame_mask: [batch, num_frames] — 1 for valid frames, 0 for padding
        """
        if attention_mask is None:
            # No padding — all frames are valid
            return torch.ones(
                hidden_states.shape[:2],
                dtype=torch.long,
                device=hidden_states.device,
            )

        # Use wav2vec2's internal method to compute output lengths
        # from the raw waveform lengths
        input_lengths = attention_mask.sum(dim=-1)  # [batch] actual waveform lengths
        output_lengths = self.wav2vec2._get_feat_extract_output_lengths(input_lengths)
        output_lengths = output_lengths.to(torch.long)

        num_frames = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]

        # Build mask: 1 for positions < output_length, 0 otherwise
        frame_indices = torch.arange(num_frames, device=hidden_states.device)
        frame_mask = (frame_indices.unsqueeze(0) < output_lengths.unsqueeze(1)).long()

        return frame_mask

    def attention_pool(self, hidden_states, frame_mask):
        """Attention-weighted mean pooling over valid frames.

        Args:
            hidden_states: [batch, num_frames, 768]
            frame_mask:    [batch, num_frames] — 1 for valid, 0 for padding

        Returns:
            pooled: [batch, 768]
        """
        attn_scores = self.attn_pool(hidden_states).squeeze(-1)   # [batch, num_frames]
        attn_scores = attn_scores.masked_fill(frame_mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)             # [batch, num_frames]

        pooled = torch.bmm(
            attn_weights.unsqueeze(1),   # [batch, 1, num_frames]
            hidden_states,               # [batch, num_frames, 768]
        ).squeeze(1)                     # [batch, 768]

        return pooled

    def forward(self, input_values, attention_mask=None):
        """
        Args:
            input_values:   [batch, raw_waveform_len]  — raw audio at 16 kHz
            attention_mask: [batch, raw_waveform_len]  — 1 for real, 0 for pad (optional)

        Returns:
            [batch, 768] pooled audio embedding
        """
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # [batch, num_frames, 768]

        frame_mask = self._compute_frame_mask(attention_mask, hidden_states)
        pooled = self.attention_pool(hidden_states, frame_mask)
        return pooled
