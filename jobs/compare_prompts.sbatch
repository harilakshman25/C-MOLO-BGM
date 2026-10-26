#!/bin/bash
#SBATCH --job-name=cmolo-prompts
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
BGM_PYTHON="${1:?Pass the absolute cmolo-gpu Python path}"
test -x "$BGM_PYTHON"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
if [[ -n "${2:-}" ]]; then
    test -f "$2"
    srun "$BGM_PYTHON" -u scripts/compare_prompts.py --adapter "$2" --out "outputs/lora-prompts-${SLURM_JOB_ID}"
else
    srun "$BGM_PYTHON" -u scripts/compare_prompts.py --out "outputs/prompts-${SLURM_JOB_ID}"
fi
