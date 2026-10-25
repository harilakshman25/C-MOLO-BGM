#!/bin/bash
#SBATCH --job-name=cmolo-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
# Submit from the repository root. Partition is selected on the sbatch command.
cd "${SLURM_SUBMIT_DIR:?Submit using sbatch from the repository root}"
printf 'Host: %s\nJob: %s\nAllocated GPU visibility: %s\n' \
  "$(hostname)" "$SLURM_JOB_ID" "${CUDA_VISIBLE_DEVICES-unset}"
printf 'Repository visible at: %s\n' "$PWD"
test -f pyproject.toml
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# Inventory only: nvidia-smi may list more than the allocated device.
# Do not set CUDA_VISIBLE_DEVICES here: Slurm supplies the allocation.
# This job does not train, install packages, or test PyTorch compatibility.
