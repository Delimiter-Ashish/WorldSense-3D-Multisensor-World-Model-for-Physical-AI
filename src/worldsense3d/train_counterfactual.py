from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import CounterfactualWorldSenseDataset
from .models import CounterfactualWorldModel
from .utils import load_yaml, resolve_device, seed_everything, write_jsonl


def weighted_loss(pred, target, mask, target_index, target_weight):
    raw = nn.functional.smooth_l1_loss(
        pred, target, reduction="none"
    ).mean(dim=-1)

    weights = mask.unsqueeze(-1).expand_as(raw).clone()
    b = torch.arange(pred.shape[0], device=pred.device)
    weights[b, target_index.long()] *= target_weight

    return (raw * weights).sum() / weights.sum().clamp_min(1)


def metrics(pred, target, factual, mask, target_index):
    dist = torch.linalg.vector_norm(pred - target, dim=-1)
    m = mask.bool().unsqueeze(-1).expand_as(dist)

    ade = dist.masked_select(m).mean()
    fde = dist[..., -1].masked_select(mask.bool()).mean()

    b = torch.arange(pred.shape[0], device=pred.device)
    td = dist[b, target_index.long()]

    true_shift = torch.linalg.vector_norm(
        target[b, target_index.long()]
        - factual[b, target_index.long()],
        dim=-1,
    )

    pred_shift = torch.linalg.vector_norm(
        pred[b, target_index.long()]
        - factual[b, target_index.long()],
        dim=-1,
    )

    return {
        "ade": ade,
        "fde": fde,
        "target_ade": td.mean(),
        "target_fde": td[:, -1].mean(),
        "shift_mae": (pred_shift - true_shift).abs().mean(),
        "final_shift_mae": (
            pred_shift[:, -1] - true_shift[:, -1]
        ).abs().mean(),
    }


def move(batch, device):
    return {
        k: v.to(device, non_blocking=True)
        for k, v in batch.items()
    }


def run_epoch(model, loader, device, cfg, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)

    names = [
        "loss", "ade", "fde", "target_ade",
        "target_fde", "shift_mae", "final_shift_mae",
    ]
    totals = {k: 0.0 for k in names}
    count = 0

    for batch in tqdm(loader, leave=False):
        batch = move(batch, device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        amp = scaler is not None and scaler.is_enabled()

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp,
        ):
            pred = model(
                batch["rgb"],
                batch["depth"],
                batch["target_rgb"],
                batch["target_depth"],
                batch["state"],
                batch["action"],
                batch["object_mask"],
            )

            loss = weighted_loss(
                pred,
                batch["target_positions"],
                batch["object_mask"],
                batch["target_index"],
                cfg["training"]["target_weight"],
            )

        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg["training"]["grad_clip"],
            )
            scaler.step(optimizer)
            scaler.update()

        m = metrics(
            pred.detach(),
            batch["target_positions"],
            batch["factual_future"],
            batch["object_mask"],
            batch["target_index"],
        )

        bs = batch["rgb"].shape[0]
        totals["loss"] += float(loss.detach()) * bs

        for k, v in m.items():
            totals[k] += float(v) * bs

        count += bs

    return {k: v / max(count, 1) for k, v in totals.items()}


def dataset(cfg, split):
    d = cfg["data"]
    return CounterfactualWorldSenseDataset(
        d["root"],
        split,
        d["context_frames"],
        d["future_frames"],
        d["depth_clip_m"],
    )


def train(config_path):
    cfg = load_yaml(config_path)
    seed_everything(int(cfg["seed"]))

    device = resolve_device(cfg["training"]["device"])
    out = Path(cfg["training"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    train_ds = dataset(cfg, cfg["data"]["train_split"])
    val_ds = dataset(cfg, cfg["data"]["val_split"])

    tc = cfg["training"]

    train_loader = DataLoader(
        train_ds,
        batch_size=tc["batch_size"],
        shuffle=True,
        num_workers=tc["num_workers"],
        pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=tc["batch_size"],
        shuffle=False,
        num_workers=tc["num_workers"],
        pin_memory=device.type == "cuda",
    )

    model = CounterfactualWorldModel(**cfg["model"]).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc["lr"],
        weight_decay=tc["weight_decay"],
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(tc["amp"] and device.type == "cuda"),
    )

    best = float("inf")

    for epoch in range(1, tc["epochs"] + 1):
        train_m = run_epoch(
            model, train_loader, device, cfg, optimizer, scaler
        )

        with torch.no_grad():
            val_m = run_epoch(
                model, val_loader, device, cfg
            )

        row = {
            "epoch": epoch,
            "train": train_m,
            "val": val_m,
        }

        print(json.dumps(row))
        write_jsonl(out / "metrics.jsonl", row)

        if val_m["target_ade"] < best:
            best = val_m["target_ade"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                },
                out / "best.pt",
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
