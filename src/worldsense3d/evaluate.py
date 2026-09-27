from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from .data import WorldSenseDataset
from .metrics import trajectory_metrics
from .models import MultiSensorWorldModel
from .utils import resolve_device


@torch.no_grad()
def evaluate(checkpoint: str, split: str = "test", batch_size: int = 32, device_name: str = "auto"):
    device = resolve_device(device_name)
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    d = cfg["data"]
    ds = WorldSenseDataset(d["root"], split, d["context_frames"], d["future_frames"], d["depth_clip_m"])
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=device.type == "cuda",
    )
    model = MultiSensorWorldModel(**cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    totals = {"ade": 0.0, "fde": 0.0, "target_ade": 0.0, "target_fde": 0.0}
    count = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(batch["rgb"], batch["depth"], batch["state"], batch["action"], batch["object_mask"])
        m = trajectory_metrics(pred, batch["target_positions"], batch["object_mask"], batch["target_index"])
        bs = batch["rgb"].shape[0]
        for k, v in m.items():
            totals[k] += float(v) * bs
        count += bs
    return {k: v / max(count, 1) for k, v in totals.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.split, args.batch_size, args.device), indent=2))


if __name__ == "__main__":
    main()
