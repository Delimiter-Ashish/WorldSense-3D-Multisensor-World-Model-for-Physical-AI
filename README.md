# WorldSense-3D

**A multisensor world model for Physical AI that learns to predict how 3D scenes evolve after physical interventions.**

WorldSense-3D is an object-centric physical reasoning system built around synchronized **RGB, depth, segmentation, structured object state, and action signals**. The first release generates controlled rigid-body interactions in PyBullet and trains a multimodal transformer to predict the **future 3D trajectory of every object in the scene**.

> Status: **v0.1 baseline implementation**. The repository contains the complete simulation → dataset → training → evaluation pipeline. Reported benchmark numbers will be added only after reproducible SCC runs are completed.

## Why this project?

Modern multimodal models are strong at describing what is visible, but Physical AI requires more: an agent must represent state, understand interventions, anticipate motion, and reason about what will happen next. WorldSense-3D turns that problem into a measurable pipeline with controlled multimodal observations and counterfactual-ready physics.

## System

```text
             ┌─────────────── Physical scene ───────────────┐
             │         objects + dynamics + push action      │
             └──────────────────────┬────────────────────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 ▼                  ▼                  ▼
               RGB(t)            Depth(t)      Object state(t)
                 │                  │                  │
          CNN + temporal     CNN + temporal        State MLP
            Transformer        Transformer             │
                 └──────────────────┬──────────────────┘
                                    ▼
                              Fusion context ◄── Action MLP
                                    │
                                    ▼
                         Object-token Transformer
                                    │
                                    ▼
                     Future 3D trajectories (x,y,z)
                         for every scene object
```

The model is deliberately structured for ablations. Any subset of `rgb`, `depth`, `state`, and `action` can be enabled in configuration, allowing direct measurement of how each sensor contributes to future-state prediction.

## Dataset episode

Each generated episode stores:

- 9 synchronized frames by default
- RGB image sequence
- metric depth sequence
- object-index segmentation masks
- padded per-object state `[position, quaternion, linear velocity, angular velocity]`
- push target and continuous action vector
- object-validity mask
- future ground-truth trajectories

The default setup uses **4 context frames + 5 prediction frames**, 3–5 randomized rigid objects, randomized mass/friction/restitution, and a planar force intervention.

## Quick start

```bash
git clone https://github.com/Delimiter-Ashish/WorldSense-3D-Multisensor-World-Model-for-Physical-AI.git
cd WorldSense-3D-Multisensor-World-Model-for-Physical-AI
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Generate a tiny smoke-test dataset:

```bash
python scripts/generate_dataset.py --config configs/base.yaml --episodes 30
```

Train:

```bash
python scripts/train.py --config configs/base.yaml
```

Evaluate the best checkpoint:

```bash
python scripts/evaluate.py --checkpoint outputs/base/best.pt --split test
```

Run tests:

```bash
pytest
```

## Metrics

WorldSense-3D v0.1 reports:

- **ADE** — mean 3D displacement error across predicted future frames
- **FDE** — final-frame 3D displacement error
- **Target ADE/FDE** — the same metrics for the directly pushed object

These metrics are computed only over valid objects; padded object slots are ignored.

## BU SCC / SLURM

The repository includes CPU dataset-generation and GPU training jobs:

```bash
sbatch slurm/generate.sbatch
sbatch slurm/train.sbatch
```

Adjust the resource directives to match the SCC partition/account available to you.

## Repository layout

```text
configs/                 experiment configuration
src/worldsense3d/sim/    PyBullet physical episode generation
src/worldsense3d/data/   multimodal dataset loader
src/worldsense3d/models/ sensor + temporal + object fusion model
src/worldsense3d/        training, evaluation, metrics and utilities
scripts/                 command-line entry points
slurm/                   SCC job scripts
tests/                   shape/data/metric regression tests
docs/ROADMAP.md          planned world-model and VLM extensions
```

## Research questions enabled by the codebase

1. How much do RGB and depth add beyond privileged object state?
2. How robust is prediction when one sensor is missing or corrupted?
3. Can a learned world model remain consistent under counterfactual interventions?
4. Which object interactions produce the largest multimodal-model failures?
5. Can a VLM use a structured dynamics model as a tool for grounded physical reasoning?

## Planned extensions

The next stages add paired counterfactual interventions, collision/contact prediction, object-centric scene graphs, sensor corruption benchmarks, and a VLM reasoning interface. See [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Reproducibility policy

No benchmark number will be placed in this README unless it is produced by the checked-in code and a recorded experiment configuration. This repository is designed to separate implemented capability from measured empirical results.

## License

MIT.
