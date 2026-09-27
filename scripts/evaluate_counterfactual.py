import argparse
import json

import torch
from torch.utils.data import DataLoader

from worldsense3d.data import CounterfactualWorldSenseDataset
from worldsense3d.models import CounterfactualWorldModel
from worldsense3d.train_counterfactual import metrics
from worldsense3d.utils import resolve_device


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]

    device = resolve_device("auto")
    d = cfg["data"]

    ds = CounterfactualWorldSenseDataset(
        d["root"],
        args.split,
        d["context_frames"],
        d["future_frames"],
        d["depth_clip_m"],
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
    )

    model = CounterfactualWorldModel(**cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    totals = {}
    per_type = {0: {}, 1: {}, 2: {}, 3: {}}
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total_count = 0

    names = ["force", "mass", "friction", "cancel_action"]

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        pred = model(
            batch["rgb"],
            batch["depth"],
            batch["target_rgb"],
            batch["target_depth"],
            batch["state"],
            batch["action"],
            batch["object_mask"],
        )

        for i in range(pred.shape[0]):
            m = metrics(
                pred[i:i+1],
                batch["target_positions"][i:i+1],
                batch["factual_future"][i:i+1],
                batch["object_mask"][i:i+1],
                batch["target_index"][i:i+1],
            )

            code = int(batch["intervention_code"][i])

            for k, v in m.items():
                value = float(v)
                totals[k] = totals.get(k, 0.0) + value
                per_type[code][k] = per_type[code].get(k, 0.0) + value

            counts[code] += 1
            total_count += 1

    overall = {
        k: v / total_count
        for k, v in totals.items()
    }

    by_intervention = {}

    for code, name in enumerate(names):
        if counts[code]:
            by_intervention[name] = {
                k: v / counts[code]
                for k, v in per_type[code].items()
            }

    print(json.dumps({
        "overall": overall,
        "by_intervention": by_intervention,
        "samples": total_count,
    }, indent=2))


if __name__ == "__main__":
    main()
