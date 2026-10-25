#!/bin/bash
#SBATCH --job-name=cmolo-lora
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
BGM_MODE="${2:-smoke}"
test -x "$BGM_PYTHON"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
srun "$BGM_PYTHON" -m unittest discover -s tests -p test_lora.py -v
case "$BGM_MODE" in
  smoke)
    srun "$BGM_PYTHON" -u scripts/train_dog_lora.py --smoke \
      --out "outputs/lora-smoke-${SLURM_JOB_ID}"
    ;;
  pilot)
    srun "$BGM_PYTHON" -u scripts/train_dog_lora.py --steps 200 \
      --out "outputs/lora-pilot-${SLURM_JOB_ID}"
    ;;
  *) printf 'Mode must be smoke or pilot\n' >&2; exit 2 ;;
esac
