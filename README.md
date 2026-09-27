# WorldSense-3D

**Intervention-aware multisensor world modeling for counterfactual Physical AI.**

WorldSense-3D is an object-centric physical reasoning system that predicts how a 3D scene evolves under **factual and counterfactual physical interventions**.

The system combines controlled PyBullet simulation with RGB, depth, segmentation, structured physical state, action information, and intervention conditioning to study a central Physical AI question:

> **If we change an action or physical property, can a learned world model predict how the future changes?**

---

## Highlights

- Paired **factual / counterfactual** physical simulations
- RGB + metric depth + segmentation + structured object state
- Future 3D trajectory prediction for every object
- Counterfactual interventions over:
  - force
  - mass
  - friction
  - action cancellation
- Target-aware object conditioning
- Intervention-effect evaluation
- Modality and intervention ablations
- 5,000-pair counterfactual benchmark
- Reproducible training and evaluation pipeline

---

## Counterfactual World Modeling

For the same initial physical scene, WorldSense-3D creates two possible futures:

```text
                     SAME INITIAL WORLD
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
          Factual World          Counterfactual World
          original physics       modified intervention
                │                       │
                ▼                       ▼
         Future trajectory       Alternative future
                │                       │
                └───────────┬───────────┘
                            ▼
                Counterfactual effect
## v0.2 Counterfactual Benchmark

WorldSense-3D v0.2 evaluates counterfactual physical reasoning on 5,000 paired factual/counterfactual scenes.

- 4,000 training pairs
- 500 validation pairs
- 500 test pairs
- 3-5 rigid objects per scene
- 4 context frames
- 5 future prediction frames
- Force, mass, friction, and action-cancellation interventions

### Ablation Results

| Model | ADE ↓ | FDE ↓ | Target ADE ↓ | Target FDE ↓ | Shift MAE ↓ | Final Shift MAE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| **State + Intervention** | **0.0361** | **0.0644** | **0.0878** | **0.1685** | **0.0588** | **0.1136** |
| Global Visual | 0.0392 | 0.0713 | 0.0994 | 0.1912 | 0.0722 | 0.1388 |
| Target-Aware Multimodal | 0.0399 | 0.0750 | 0.0991 | 0.1925 | 0.0693 | 0.1330 |
| No Intervention | 0.0848 | 0.1483 | 0.2898 | 0.5038 | 0.1495 | 0.2527 |

## Main Findings

Explicit intervention conditioning substantially improves counterfactual forecasting.

Compared with the no-intervention model, the strongest intervention-aware model reduces:

- Target ADE by approximately **69.7%**
- Counterfactual shift MAE by approximately **60.7%**

Structured state + intervention conditioning achieves the strongest result in the current simulator.

RGB and depth do not outperform privileged structured physical state in the current setup. This motivates future experiments where state information is incomplete, noisy, or must be inferred directly from perception.

Mass interventions are the hardest tested intervention, while action cancellation is the easiest.

## Benchmark Figures

### Counterfactual Target Prediction

![Target ADE](docs/figures/v02_target_ade.png)

### Intervention-Effect Prediction

![Shift MAE](docs/figures/v02_shift_mae.png)

### Intervention Difficulty

![Intervention Breakdown](docs/figures/v02_intervention_breakdown.png)

## Qualitative Counterfactual Futures

![Counterfactual Overview](docs/figures/counterfactual_examples/counterfactual_overview.png)

Additional factual-vs-counterfactual animations and trajectory plots are available in:

docs/figures/counterfactual_examples/

## Metrics

- **ADE** - average 3D displacement error over future frames
- **FDE** - final-frame 3D displacement error
- **Target ADE/FDE** - trajectory error for the intervened object
- **Shift MAE** - error in the magnitude of intervention-induced trajectory change
- **Final Shift MAE** - intervention-effect error at the final prediction frame

## Dataset Contents

Each counterfactual pair contains:

- RGB sequence
- metric depth sequence
- semantic segmentation
- structured object states
- object validity mask
- target object index
- factual and counterfactual actions
- target mass and friction
- effective force
- intervention type and scale
- factual future trajectories
- counterfactual future trajectories

Structured object state includes 3D position, quaternion orientation, linear velocity, and angular velocity.

## Quick Start

Clone the repository:

    git clone git@github.com:Delimiter-Ashish/WorldSense-3D-Multisensor-World-Model-for-Physical-AI.git
    cd WorldSense-3D-Multisensor-World-Model-for-Physical-AI
    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"

Generate counterfactual data:

    python scripts/generate_counterfactual_dataset.py --config configs/counterfactual.yaml --pairs 5000

Train:

    python scripts/train_counterfactual.py --config configs/counterfactual_5000.yaml

Evaluate:

    python scripts/evaluate_counterfactual.py --checkpoint outputs/counterfactual_5000/best.pt --split test

Run tests:

    pytest -q

## Repository Structure

- `configs/` - experiment configurations
- `src/worldsense3d/sim/` - factual and counterfactual PyBullet simulation
- `src/worldsense3d/data/` - dataset loaders
- `src/worldsense3d/models/` - world-model architectures
- `scripts/` - generation, training, evaluation, and visualization
- `docs/` - benchmark reports and figures
- `results/` - recorded benchmark summaries
- `tests/` - regression and shape tests

## Research Questions

1. Can world models predict alternative physical futures?
2. How important is explicit intervention conditioning?
3. Which physical interventions are hardest to model?
4. When does vision add information beyond privileged state?
5. Can physical properties be inferred directly from perception?
6. How robust are world models to missing or corrupted sensors?

## Future Directions

- vision-only and partial-state world modeling
- hidden physical-property inference
- object-centric visual representations
- collision and contact prediction
- sensor corruption benchmarks
- uncertainty-aware future prediction
- VLM/world-model integration
- embodied-agent planning

## Reproducibility

Generated datasets, virtual environments, checkpoints, and raw training outputs are excluded from Git.

Tracked artifacts include source code, configurations, benchmark summaries, figures, tests, and documentation.

## License

MIT
