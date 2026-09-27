# WorldSense-3D Roadmap

## v0.1 — Multisensor dynamics baseline
- PyBullet RGB/depth/segmentation/state/action episode generator
- Full-object future 3D trajectory prediction
- RGB/depth/state/action fusion transformer
- ADE/FDE and pushed-object metrics
- SCC/SLURM reproducibility

## v0.2 — Counterfactual physical reasoning
- Paired interventions with altered force, mass and friction
- Counterfactual consistency metrics
- Contact/collision prediction head
- Object-centric scene graphs

## v0.3 — Foundation-model interface
- Natural-language physical questions grounded in simulated episodes
- VLM adapters for Qwen/InternVL/LLaVA-family models
- Structured tool calls into the learned dynamics model
- Evidence-grounded answers with uncertainty

## v0.4 — Interpretability and robustness
- Sensor-dropout and corruption benchmark
- Object masking and modality ablation
- Failure taxonomy by geometry, contact and occlusion
- Causal intervention visualizations

## v1.0 — Physical AI world-model benchmark
- Large-scale dataset release
- Strong learned and analytical baselines
- Reproducible benchmark suite
- Interactive demo and model cards
