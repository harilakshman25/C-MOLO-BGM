#!/bin/bash
#SBATCH --job-name=cmolo-baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
BGM_PYTHON="${1:?Pass the absolute cmolo-gpu Python path}"
test -x "$BGM_PYTHON"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
printf 'Host: %s\nJob: %s\n' "$(hostname)" "$SLURM_JOB_ID"
# Slurm controls device visibility. Model download must finish before submission.
srun "$BGM_PYTHON" -u scripts/run_baseline.py \
  --steps 100 --out "outputs/baseline-${SLURM_JOB_ID}"
