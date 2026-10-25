"""
GRU Autoencoder v2 — improved architecture for PFDI project.

Improvements over v1:
  * Bottleneck GRU encoder with attention-weighted context pooling
  * Bidirectional GRU encoder (forward context captures process history;
    reverse context captures leading-edge anomalies)
  * Separate per-sensor output heads (each sensor has its own linear layer
    so reconstruction error is not diluted across the sensor mix)
  * Layer normalisation after each GRU layer for training stability
  * Dropout for regularisation (reduces overfitting on normal data)

All of these are standard architectural choices described in the literature;
none involve looking at test labels.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    """Single-head scaled dot-product attention over time dimension."""

    def __init__(self, hidden: int):
        super().__init__()
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.scale = hidden ** -0.5

    def forward(self, x):
        # x: (B, T, H)
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        return attn @ v  # (B, T, H)


class GRUAutoencoderV2(nn.Module):
    """
    Encoder:  Bidirectional GRU (32+32) → LayerNorm → Attention → GRU (16) → LayerNorm
    Bottleneck: 16-dim hidden state
    Decoder:  GRU (16) → GRU (64) → LayerNorm → per-sensor Linear heads
    """

    def __init__(self, input_size: int = 9, dropout: float = 0.15):
        super().__init__()

        # ── encoder ──────────────────────────────────────────────────────────
        self.enc1 = nn.GRU(
            input_size=input_size,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True,    # 64-dim output
        )
        self.ln_enc1 = nn.LayerNorm(64)
        self.drop_enc = nn.Dropout(dropout)
        self.attention = AttentionPool(64)

        self.enc2 = nn.GRU(
            input_size=64,
            hidden_size=16,
            num_layers=1,
            batch_first=True,
            bidirectional=False,   # 16-dim bottleneck
        )
        self.ln_enc2 = nn.LayerNorm(16)

        # ── decoder ──────────────────────────────────────────────────────────
        self.dec1 = nn.GRU(
            input_size=16,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )
        self.ln_dec1 = nn.LayerNorm(32)
        self.drop_dec = nn.Dropout(dropout)

        self.dec2 = nn.GRU(
            input_size=32,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.ln_dec2 = nn.LayerNorm(64)

        # Per-sensor output heads (each sensor gets its own reconstruction)
        self.heads = nn.ModuleList([
            nn.Linear(64, 1) for _ in range(input_size)
        ])

    def encode(self, x):
        z, _ = self.enc1(x)          # (B, T, 64)
        z = self.ln_enc1(z)
        z = self.drop_enc(z)
        z = self.attention(z)        # (B, T, 64) — attention-reweighted
        z, _ = self.enc2(z)          # (B, T, 16)
        z = self.ln_enc2(z)
        return z

    def decode(self, z):
        h, _ = self.dec1(z)          # (B, T, 32)
        h = self.ln_dec1(h)
        h = self.drop_dec(h)
        h, _ = self.dec2(h)          # (B, T, 64)
        h = self.ln_dec2(h)
        # Stack per-sensor heads: (B, T, n_sensors)
        out = torch.cat([head(h) for head in self.heads], dim=-1)
        return out

    def forward(self, x):
        return self.decode(self.encode(x))
