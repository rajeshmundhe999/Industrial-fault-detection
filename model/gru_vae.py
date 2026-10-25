"""
GRU Variational Autoencoder (GRU-VAE)
======================================
An upgrade over GRU-AE v2. The key differences:

1. The encoder outputs a DISTRIBUTION (μ, σ²) over the latent space,
   not a fixed point. This is the VAE latent variable model.

2. The anomaly score combines:
   - Reconstruction error (per-sensor, same as v2)
   - KL divergence from N(0,1) prior — penalises unusual process states
     even when reconstruction is low. This catches subtle distributional shifts.

3. Uncertainty is principled: σ² from the encoder IS the model's uncertainty.
   No MC-Dropout needed — one forward pass gives both score and confidence.

4. The structured Gaussian prior on the latent space makes the fault isolation
   clustering (from fault_isolation_clustering.py) significantly more meaningful.

Architecture (same parameter budget as v2, ~50k params)
---------------------------------------------------------
Encoder:
  BiGRU(input_size → 64) → LayerNorm → Dropout → AttentionPool
  → GRU(64 → 32) → LayerNorm
  → Linear(32 → 16)  [μ head]
  → Linear(32 → 16)  [logvar head]

Reparameterisation:
  z = μ + ε·exp(0.5·logvar),  ε ~ N(0,I)   [training]
  z = μ                                       [inference]

Decoder:
  GRU(16 → 32) → LayerNorm → Dropout
  → GRU(32 → 64) → LayerNorm
  → 9 × Linear(64 → 1)  [per-sensor output heads]
"""
import torch
import torch.nn as nn


class _AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, h):
        # h: (B, T, D)
        w = torch.softmax(self.attn(h), dim=1)   # (B, T, 1)
        ctx = (w * h).sum(dim=1, keepdim=True)    # (B, 1, D)
        return ctx.expand(-1, h.size(1), -1)       # (B, T, D)


class GRUVAEEncoder(nn.Module):
    def __init__(self, input_size, hidden=64, latent_dim=16, dropout=0.15):
        super().__init__()
        self.bigru   = nn.GRU(input_size, hidden, batch_first=True, bidirectional=True)
        self.norm1   = nn.LayerNorm(hidden * 2)
        self.drop    = nn.Dropout(dropout)
        self.attn    = _AttentionPool(hidden * 2)
        self.gru2    = nn.GRU(hidden * 2, latent_dim * 2, batch_first=True)
        self.norm2   = nn.LayerNorm(latent_dim * 2)
        self.fc_mu   = nn.Linear(latent_dim * 2, latent_dim)
        self.fc_logv = nn.Linear(latent_dim * 2, latent_dim)

    def forward(self, x):
        h, _  = self.bigru(x)           # (B, T, 2H)
        h     = self.drop(self.norm1(h))
        ctx   = self.attn(h)            # (B, T, 2H)
        h2, _ = self.gru2(ctx)         # (B, T, 2·latent)
        h2    = self.norm2(h2)
        hT    = h2[:, -1, :]           # (B, 2·latent) — last time step
        mu    = self.fc_mu(hT)         # (B, latent)
        logv  = self.fc_logv(hT)       # (B, latent)
        # clamp logvar to prevent numerical instability
        logv  = torch.clamp(logv, -10, 4)
        return mu, logv

    def encode(self, x):
        """Deterministic encode (returns μ only — for clustering/plotting)."""
        mu, _ = self.forward(x)
        return mu


class GRUVAEDecoder(nn.Module):
    def __init__(self, latent_dim=16, hidden=64, output_size=9, dropout=0.15):
        super().__init__()
        self.gru1   = nn.GRU(latent_dim, 32, batch_first=True)
        self.norm1  = nn.LayerNorm(32)
        self.drop   = nn.Dropout(dropout)
        self.gru2   = nn.GRU(32, hidden, batch_first=True)
        self.norm2  = nn.LayerNorm(hidden)
        # Per-sensor output heads — one linear layer per sensor
        self.heads  = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(output_size)])

    def forward(self, z, seq_len):
        # Expand latent vector across time → (B, T, latent_dim)
        z_seq      = z.unsqueeze(1).expand(-1, seq_len, -1)
        h1, _      = self.gru1(z_seq)           # (B, T, 32)
        h1         = self.drop(self.norm1(h1))
        h2, _      = self.gru2(h1)              # (B, T, hidden)
        h2         = self.norm2(h2)
        # Apply per-sensor heads and concatenate
        outs       = [head(h2) for head in self.heads]  # list of (B, T, 1)
        return torch.cat(outs, dim=-1)           # (B, T, output_size)


class GRUVAEModel(nn.Module):
    """
    Full GRU-VAE model.

    Anomaly score for a window x:
      score(x) = robust_max(sensor_recon_errors) + beta * KL(q(z|x) || N(0,I))

    The KL term catches process states that are unusual in latent space even
    when individual sensor reconstructions are acceptable.
    """
    def __init__(self, input_size=9, hidden=64, latent_dim=16, dropout=0.15):
        super().__init__()
        self.input_size  = input_size
        self.latent_dim  = latent_dim
        self.encoder     = GRUVAEEncoder(input_size, hidden, latent_dim, dropout)
        self.decoder     = GRUVAEDecoder(latent_dim, hidden, input_size, dropout)

    def reparameterise(self, mu, logv):
        if self.training:
            std = torch.exp(0.5 * logv)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # deterministic at test time

    def forward(self, x):
        """
        Returns:
            recon   (B, T, S) — reconstructed input
            mu      (B, latent_dim)
            logv    (B, latent_dim)
        """
        mu, logv = self.encoder(x)
        z        = self.reparameterise(mu, logv)
        recon    = self.decoder(z, x.size(1))
        return recon, mu, logv

    def encode(self, x):
        """Deterministic μ for downstream clustering."""
        return self.encoder.encode(x)

    @staticmethod
    def kl_divergence(mu, logv):
        """
        KL(q(z|x) || N(0,I)) per sample (sum over latent dims).
        = -0.5 * sum(1 + logvar - mu² - exp(logvar))
        """
        return -0.5 * torch.sum(1 + logv - mu.pow(2) - logv.exp(), dim=1)  # (B,)

    def elbo_loss(self, x, recon, mu, logv, beta=1.0):
        """
        Negative ELBO = reconstruction_loss + beta * KL
        reconstruction_loss = MSE per element (mean over T and S)
        """
        recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
        kl         = self.kl_divergence(mu, logv).mean()
        return recon_loss + beta * kl, recon_loss.item(), kl.item()
