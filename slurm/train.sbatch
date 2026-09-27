#!/bin/bash
#SBATCH --job-name=ws3d-train
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs
source .venv/bin/activate
nvidia-smi
python scripts/train.py --config configs/base.yaml
