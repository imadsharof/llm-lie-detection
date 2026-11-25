#!/bin/bash
# CPU job: train/evaluate the 8 probes and the black-box classifier, one dataset per task.
# Usage:  sbatch jobs/train_eval.sbatch     (after the extract job has finished)
#SBATCH --job-name=train_eval
#SBATCH --array=0-14
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_eval_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load Python/3.11.3-GCCcore-12.3.0
source .venv/bin/activate

python scripts/train_eval_blackbox.py
python scripts/train_eval_whitebox.py