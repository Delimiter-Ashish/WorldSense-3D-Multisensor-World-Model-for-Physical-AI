from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import WorldSenseDataset
from .metrics import trajectory_metrics
from .models import MultiSensorWorldModel
from .utils import load_yaml, resolve_device, seed_everything, write_jsonl


def masked_smooth_l1(
    pred: torch.Tensor, target: torch.Tensor, object_mask: torch.Tensor
) -> torch.Tensor:
    raw = nn.functional.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)
    mask = object_mask.bool().unsqueeze(-1).expand_as(raw)
    return raw.masked_select(mask).mean()


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def run_epoch(model, loader, device, optimizer=None, scaler=None, grad_clip=1.0):
    training = optimizer is not None
    model.train(training)
    sums = {"loss": 0.0, "ade": 0.0, "fde": 0.0, "target_ade": 0.0, "target_fde": 0.0}
    count = 0
    amp_enabled = scaler is not None and scaler.is_enabled()

    for batch in tqdm(loader, leave=False):
        batch = move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            pred = model(
                batch["rgb"], batch["depth"], batch["state"], batch["action"], batch["object_mask"]
            )
            loss = masked_smooth_l1(pred, batch["target_positions"], batch["object_mask"])
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

        metrics = trajectory_metrics(
            pred.detach(), batch["target_positions"], batch["object_mask"], batch["target_index"]
        )
        bs = batch["rgb"].shape[0]
        sums["loss"] += float(loss.detach()) * bs
        for k, v in metrics.items():
            sums[k] += float(v) * bs
        count += bs
    return {k: v / max(count, 1) for k, v in sums.items()}


def build_dataset(cfg, split):
    d = cfg["data"]
    return WorldSenseDataset(
        d["root"], split, d["context_frames"], d["future_frames"], d["depth_clip_m"]
    )


def train(config_path: str) -> Path:
    cfg = load_yaml(config_path)
    seed_everything(int(cfg["seed"]))
    device = resolve_device(cfg["training"]["device"])
    out = Path(cfg["training"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    train_ds = build_dataset(cfg, cfg["data"]["train_split"])
    val_ds = build_dataset(cfg, cfg["data"]["val_split"])
    tc = cfg["training"]
    train_loader = DataLoader(
        train_ds, batch_size=tc["batch_size"], shuffle=True, num_workers=tc["num_workers"], pin_memory=device.type == "cuda"
    )
    val_loader = DataLoader(
        val_ds, batch_size=tc["batch_size"], shuffle=False, num_workers=tc["num_workers"], pin_memory=device.type == "cuda"
    )

    model = MultiSensorWorldModel(**cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=bool(tc["amp"] and device.type == "cuda"))

    best = float("inf")
    for epoch in range(1, tc["epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, scaler, tc["grad_clip"])
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, device)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        write_jsonl(out / "metrics.jsonl", row)
        print(json.dumps(row))
        if val_metrics["ade"] < best:
            best = val_metrics["ade"]
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch}, out / "best.pt")
    return out / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
