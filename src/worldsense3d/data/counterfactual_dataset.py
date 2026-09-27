from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CounterfactualWorldSenseDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        context_frames: int = 4,
        future_frames: int = 5,
        depth_clip_m: float = 5.0,
    ) -> None:
        self.root = Path(root) / split
        self.files = sorted(self.root.glob("pair_*.npz"))

        if not self.files:
            raise FileNotFoundError(f"No counterfactual pairs under {self.root}")

        self.context_frames = context_frames
        self.future_frames = future_frames
        self.depth_clip_m = depth_clip_m

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        with np.load(self.files[index]) as d:
            rgb = d["factual_rgb"][: self.context_frames].astype(np.float32) / 255.0
            depth = d["factual_depth"][: self.context_frames].astype(np.float32)
            segmentation = d["factual_segmentation"][: self.context_frames]

            factual_states = d["factual_states"].astype(np.float32)
            counterfactual_states = d["counterfactual_states"].astype(np.float32)

            object_mask = d["factual_object_mask"].astype(np.float32)
            target_index = int(d["target_index"])

            intervention_code = int(d["intervention_code"])
            intervention_scale = float(d["intervention_scale"])

            factual_action = d["factual_action"].astype(np.float32)
            counterfactual_action = d["counterfactual_action"].astype(np.float32)

            factual_phys = np.asarray(
                [
                    float(d["factual_target_mass"]),
                    float(d["factual_target_friction"]),
                    float(d["factual_effective_force"]),
                ],
                dtype=np.float32,
            )

            counterfactual_phys = np.asarray(
                [
                    float(d["counterfactual_target_mass"]),
                    float(d["counterfactual_target_friction"]),
                    float(d["counterfactual_effective_force"]),
                ],
                dtype=np.float32,
            )

        depth = np.clip(depth / self.depth_clip_m, 0.0, 1.0)

        target_mask = (segmentation == (target_index + 1)).astype(np.float32)
        target_rgb = rgb * target_mask[..., None]
        target_depth = depth * target_mask

        rgb = np.transpose(rgb, (0, 3, 1, 2))
        target_rgb = np.transpose(target_rgb, (0, 3, 1, 2))
        depth = depth[:, None]
        target_depth = target_depth[:, None]

        state = factual_states[self.context_frames - 1]

        target_flag = np.zeros((state.shape[0], 1), dtype=np.float32)
        target_flag[target_index, 0] = 1.0
        state = np.concatenate([state, target_flag], axis=-1)

        one_hot = np.zeros(4, dtype=np.float32)
        one_hot[intervention_code] = 1.0

        condition = np.concatenate(
            [
                counterfactual_action,
                one_hot,
                np.asarray([intervention_scale], dtype=np.float32),
                factual_phys,
                counterfactual_phys,
            ]
        ).astype(np.float32)

        target_positions = counterfactual_states[
            self.context_frames :
            self.context_frames + self.future_frames,
            :,
            :3,
        ]
        target_positions = np.transpose(target_positions, (1, 0, 2))

        factual_future = factual_states[
            self.context_frames :
            self.context_frames + self.future_frames,
            :,
            :3,
        ]
        factual_future = np.transpose(factual_future, (1, 0, 2))

        return {
            "rgb": torch.from_numpy(rgb),
            "depth": torch.from_numpy(depth),
            "target_rgb": torch.from_numpy(target_rgb),
            "target_depth": torch.from_numpy(target_depth),
            "state": torch.from_numpy(state),
            "action": torch.from_numpy(condition),
            "object_mask": torch.from_numpy(object_mask),
            "target_positions": torch.from_numpy(target_positions),
            "factual_future": torch.from_numpy(factual_future),
            "target_index": torch.tensor(target_index, dtype=torch.long),
            "intervention_code": torch.tensor(intervention_code, dtype=torch.long),
        }
