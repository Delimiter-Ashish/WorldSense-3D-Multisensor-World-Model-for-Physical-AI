from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class WorldSenseDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        context_frames: int = 4,
        future_frames: int = 5,
        depth_clip_m: float = 5.0,
    ) -> None:
        self.root = Path(root) / split
        self.files = sorted(self.root.glob("episode_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No episodes found under {self.root}")
        self.context_frames = context_frames
        self.future_frames = future_frames
        self.depth_clip_m = depth_clip_m

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        with np.load(self.files[index]) as d:
            rgb = d["rgb"][: self.context_frames].astype(np.float32) / 255.0
            depth = d["depth"][: self.context_frames].astype(np.float32)
            states = d["states"].astype(np.float32)
            action = d["action"].astype(np.float32)
            object_mask = d["object_mask"].astype(np.float32)
            target_index = int(d["target_index"])

        depth = np.clip(depth / self.depth_clip_m, 0.0, 1.0)
        rgb = np.transpose(rgb, (0, 3, 1, 2))
        depth = depth[:, None]

        last_context = states[self.context_frames - 1]
        target_positions = states[
            self.context_frames : self.context_frames + self.future_frames, :, :3
        ]
        target_positions = np.transpose(target_positions, (1, 0, 2))

        return {
            "rgb": torch.from_numpy(rgb),
            "depth": torch.from_numpy(depth),
            "state": torch.from_numpy(last_context),
            "action": torch.from_numpy(action),
            "object_mask": torch.from_numpy(object_mask),
            "target_positions": torch.from_numpy(target_positions),
            "target_index": torch.tensor(target_index, dtype=torch.long),
        }
