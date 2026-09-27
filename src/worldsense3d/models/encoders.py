from __future__ import annotations

import torch
from torch import nn


class ConvFrameEncoder(nn.Module):
    def __init__(self, in_channels: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(96, out_dim, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


class TemporalVisualEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        visual_dim: int,
        hidden_dim: int,
        layers: int,
        heads: int,
        dropout: float,
        max_frames: int = 16,
    ) -> None:
        super().__init__()
        self.frame_encoder = ConvFrameEncoder(in_channels, visual_dim)
        self.proj = nn.Linear(visual_dim, hidden_dim)
        self.pos = nn.Parameter(torch.randn(1, max_frames, hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        frames = self.frame_encoder(x.reshape(b * t, c, h, w)).reshape(b, t, -1)
        tokens = self.proj(frames) + self.pos[:, :t]
        tokens = self.temporal(tokens)
        return self.norm(tokens.mean(dim=1))
