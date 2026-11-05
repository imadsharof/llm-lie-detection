#!/bin/bash
#SBATCH --job-name=test-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --output=test_%j.out

module load Python
source ~/venv/bin/activate
python slurm/test_gpu.py