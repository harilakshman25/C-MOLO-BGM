#!/bin/bash
#SBATCH --job-name=cmolo-compose
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
BGM_PYTHON="${1:?Pass absolute environment Python path}"
BGM_ADAPTER="${2:?Pass trained adapter path}"
shift 2
test -x "$BGM_PYTHON"
test -f "$BGM_ADAPTER"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
srun "$BGM_PYTHON" -m unittest discover -s tests -p test_window_composition.py -v
srun "$BGM_PYTHON" -u scripts/run_composition_study.py --adapter "$BGM_ADAPTER" --out "outputs/composition-${SLURM_JOB_ID}" "$@"
