#!/bin/bash
#SBATCH --job-name=ws3d-generate
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs
source .venv/bin/activate
python scripts/generate_dataset.py --config configs/base.yaml --episodes 10000
