#!/bin/bash
#SBATCH --job-name=cmolo-regional
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
BGM_PYTHON="${1:?Pass absolute GPU environment Python path}"
BGM_ADAPTER="${2:?Pass trained adapter path}"
test -x "$BGM_PYTHON"
test -f "$BGM_ADAPTER"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
srun "$BGM_PYTHON" -u scripts/compare_regional.py --adapter "$BGM_ADAPTER" --out "outputs/regional-${SLURM_JOB_ID}"
