from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import trange

from worldsense3d.sim import EpisodeGenerator, SimulationConfig
from worldsense3d.utils import load_yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    sim_cfg = dict(cfg["simulation"])
    episodes = int(args.episodes or sim_cfg.pop("episodes"))
    train_fraction = float(sim_cfg.pop("train_fraction"))
    val_fraction = float(sim_cfg.pop("val_fraction"))
    out = Path(args.output or cfg["data"]["root"])
    gen = EpisodeGenerator(SimulationConfig(**sim_cfg), seed=int(cfg["seed"]))

    rng = np.random.default_rng(int(cfg["seed"]))
    order = rng.permutation(episodes)
    n_train = int(episodes * train_fraction)
    n_val = int(episodes * val_fraction)
    split_for = {}
    for rank, idx in enumerate(order):
        split_for[int(idx)] = "train" if rank < n_train else "val" if rank < n_train + n_val else "test"

    try:
        for i in trange(episodes):
            split = split_for[i]
            gen.save_episode(out / split / f"episode_{i:06d}.npz")
    finally:
        gen.close()


if __name__ == "__main__":
    main()
