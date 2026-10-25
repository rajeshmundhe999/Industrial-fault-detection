"""
LSTM-AE Baseline Model
======================
The simplest possible LSTM autoencoder — the standard comparison point
against which GRU-AE v2 and GRU-VAE are measured.

Architecture:
  Encoder: LSTM(9 → 32) → take last hidden state
  Decoder: repeat hidden state T times → LSTM(32 → 32) → Linear(32 → 9)

Deliberately simple (~12k parameters vs v2's 49k) to show that the
improvements in our architecture are meaningful, not just from scale.
"""
import torch
import torch.nn as nn


class LSTMAEModel(nn.Module):
    """
    Simple LSTM autoencoder for multivariate time series.
    Encoder compresses T×S → latent_dim.
    Decoder reconstructs latent_dim → T×S.
    """
    def __init__(self, input_size=9, hidden=32, latent_dim=8, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim

        # Encoder
        self.enc_lstm = nn.LSTM(input_size, hidden, batch_first=True,
                                num_layers=1, dropout=0.0)
        self.enc_fc   = nn.Linear(hidden, latent_dim)

        # Decoder
        self.dec_fc   = nn.Linear(latent_dim, hidden)
        self.dec_lstm = nn.LSTM(hidden, hidden, batch_first=True,
                                num_layers=1, dropout=0.0)
        self.out      = nn.Linear(hidden, input_size)
        self.drop     = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape

        # Encode
        _, (h, _) = self.enc_lstm(x)       # h: (1, B, hidden)
        h         = h.squeeze(0)            # (B, hidden)
        z         = self.enc_fc(h)          # (B, latent_dim)

        # Decode — expand z across time
        z_exp  = self.dec_fc(z).unsqueeze(1).expand(-1, T, -1)  # (B, T, hidden)
        d, _   = self.dec_lstm(z_exp)       # (B, T, hidden)
        d      = self.drop(d)
        return self.out(d)                  # (B, T, input_size)

    def encode(self, x):
        """Returns latent vector (for clustering)."""
        _, (h, _) = self.enc_lstm(x)
        return self.enc_fc(h.squeeze(0))
