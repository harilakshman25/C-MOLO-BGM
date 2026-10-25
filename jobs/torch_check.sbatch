#!/bin/bash
#SBATCH --job-name=cmolo-torch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
BGM_PYTHON="${1:?Pass the absolute cmolo-gpu Python path as the first script argument}"
test -x "$BGM_PYTHON"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Keep Slurm's CUDA_VISIBLE_DEVICES unchanged.
srun "$BGM_PYTHON" -u scripts/check_gpu.py
