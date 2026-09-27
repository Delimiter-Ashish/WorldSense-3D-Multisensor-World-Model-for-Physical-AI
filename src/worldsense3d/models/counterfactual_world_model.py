from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from .encoders import TemporalVisualEncoder


class CounterfactualWorldModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 192,
        visual_dim: int = 128,
        temporal_layers: int = 2,
        scene_layers: int = 3,
        num_heads: int = 6,
        dropout: float = 0.1,
        state_dim: int = 14,
        action_dim: int = 17,
        max_objects: int = 5,
        future_frames: int = 5,
        modalities: Iterable[str] = (
            "rgb", "depth", "target_rgb", "target_depth", "state", "action"
        ),
    ) -> None:
        super().__init__()
        self.modalities = set(modalities)
        self.hidden_dim = hidden_dim
        self.max_objects = max_objects
        self.future_frames = future_frames

        self.rgb_encoder = TemporalVisualEncoder(
            3, visual_dim, hidden_dim, temporal_layers, num_heads, dropout
        )
        self.depth_encoder = TemporalVisualEncoder(
            1, visual_dim, hidden_dim, temporal_layers, num_heads, dropout
        )
        self.target_rgb_encoder = TemporalVisualEncoder(
            3, visual_dim, hidden_dim, temporal_layers, num_heads, dropout
        )
        self.target_depth_encoder = TemporalVisualEncoder(
            1, visual_dim, hidden_dim, temporal_layers, num_heads, dropout
        )

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.object_pos = nn.Parameter(
            torch.randn(1, max_objects, hidden_dim) * 0.02
        )

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.scene = nn.TransformerEncoder(layer, num_layers=scene_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, future_frames * 3),
        )

    def forward(
        self,
        rgb,
        depth,
        target_rgb,
        target_depth,
        state,
        action,
        object_mask,
    ):
        b, n, _ = state.shape

        global_context = torch.zeros(
            b, self.hidden_dim, device=state.device, dtype=state.dtype
        )

        if "rgb" in self.modalities:
            global_context += self.rgb_encoder(rgb)

        if "depth" in self.modalities:
            global_context += self.depth_encoder(depth)

        if "action" in self.modalities:
            global_context += self.action_encoder(action)

        target_context = torch.zeros_like(global_context)

        if "target_rgb" in self.modalities:
            target_context += self.target_rgb_encoder(target_rgb)

        if "target_depth" in self.modalities:
            target_context += self.target_depth_encoder(target_depth)

        if "state" in self.modalities:
            tokens = self.state_encoder(state)
        else:
            tokens = torch.zeros(
                b, n, self.hidden_dim,
                device=state.device,
                dtype=state.dtype,
            )

        target_flag = state[..., -1:].clamp(0.0, 1.0)

        tokens = (
            tokens
            + global_context[:, None, :]
            + target_flag * target_context[:, None, :]
            + self.object_pos[:, :n]
        )

        tokens = self.scene(
            tokens,
            src_key_padding_mask=~object_mask.bool(),
        )

        tokens = self.norm(tokens)

        delta = self.head(tokens).reshape(
            b, n, self.future_frames, 3
        )

        return state[..., :3].unsqueeze(2) + delta
