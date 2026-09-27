# WorldSense-3D v0.1 Experiments

## Environment
- BU SCC
- NVIDIA A100-SXM4-80GB
- Python 3.10
- PyTorch + CUDA
- PyBullet simulation

## Dataset
5,000 synthetic physical interaction episodes:
- 4,000 train
- 500 validation
- 500 test

Each episode contains:
- RGB
- depth
- segmentation
- structured object state
- applied action
- future 3D trajectories

## Final v0.1 Results

| Model | ADE | FDE | Target ADE | Target FDE |
|---|---:|---:|---:|---:|
| RGB + Depth + State + Action | 0.109949 | 0.171377 | 0.338759 | 0.574092 |
| State + Action | 0.119725 | 0.204891 | 0.267285 | 0.463157 |

## Observation
Multimodal sensing improves overall scene trajectory prediction, while the
state-action baseline performs better on the directly intervened target object.
This motivates target-aware visual grounding and counterfactual reasoning in v0.2.

## v0.2 Direction
- target-aware object grounding
- paired factual/counterfactual scenes
- force interventions
- mass interventions
- friction interventions
- counterfactual trajectory metrics
