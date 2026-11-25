#!/bin/bash
# GPU job: white-box activations + black-box features, one dataset per array task.
# Usage (from the project root):  sbatch jobs/extract.sbatch
#SBATCH --job-name=extract
#SBATCH --array=0-14               # one task per dataset of configs/default.yaml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/extract_%A_%a.out
##SBATCH --partition=...           # check the GPU partition name on Lyra

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

export HF_HOME=$HOME/hf_cache       # where the model was downloaded beforehand
export HF_HUB_OFFLINE=1             # compute nodes usually have no internet
export HF_DATASETS_OFFLINE=1

python scripts/extract_whitebox.py
python scripts/extract_blackbox.py