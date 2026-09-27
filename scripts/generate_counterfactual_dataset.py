from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import trange

from worldsense3d.sim.counterfactual import CounterfactualEpisodeGenerator
from worldsense3d.sim import SimulationConfig
from worldsense3d.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/counterfactual.yaml")
    parser.add_argument("--pairs", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    sim_cfg = dict(cfg["simulation"])
    default_pairs = int(sim_cfg.pop("pairs"))
    pairs = int(args.pairs if args.pairs is not None else default_pairs)

    train_fraction = float(sim_cfg.pop("train_fraction"))
    val_fraction = float(sim_cfg.pop("val_fraction"))

    out = Path(args.output or cfg["data"]["root"])

    generator = CounterfactualEpisodeGenerator(
        SimulationConfig(**sim_cfg),
        seed=int(cfg["seed"]),
    )

    rng = np.random.default_rng(int(cfg["seed"]))
    order = rng.permutation(pairs)

    n_train = int(pairs * train_fraction)
    n_val = int(pairs * val_fraction)

    split_for = {}

    for rank, idx in enumerate(order):
        if rank < n_train:
            split = "train"
        elif rank < n_train + n_val:
            split = "val"
        else:
            split = "test"

        split_for[int(idx)] = split

    try:
        for i in trange(pairs):
            split = split_for[i]
            generator.save_pair(
                out / split / f"pair_{i:06d}.npz"
            )
    finally:
        generator.close()


if __name__ == "__main__":
    main()
