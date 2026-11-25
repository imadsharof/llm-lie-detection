#!/bin/bash
# Quick end-to-end check on ONE small dataset before launching everything.
#SBATCH --job-name=test_small
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/test_small_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load Python/3.11.3-GCCcore-12.3.0
source .venv/bin/activate
export HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

python scripts/extract_whitebox.py --dataset got_sp_en_trans
python scripts/extract_blackbox.py --dataset got_sp_en_trans
python scripts/train_eval_blackbox.py --dataset got_sp_en_trans
python scripts/train_eval_whitebox.py --dataset got_sp_en_trans --methods lr lr-g dim