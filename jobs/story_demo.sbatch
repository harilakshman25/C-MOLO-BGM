#!/bin/bash
#SBATCH --job-name=cmolo-story
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
BGM_PYTHON="${1:?Pass absolute GPU Python path}"
BGM_ADAPTER="${2:?Pass trained dog adapter}"
BGM_NARRATION="${3:-outputs/forest-narration}"
test -f "$BGM_NARRATION/narration.wav"
test -x "$BGM_PYTHON"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
srun "$BGM_PYTHON" -u scripts/generate_library.py --adapter "$BGM_ADAPTER" --out "outputs/sound-library-${SLURM_JOB_ID}"
srun "$BGM_PYTHON" -u scripts/render_story.py --narration "$BGM_NARRATION" --library "outputs/sound-library-${SLURM_JOB_ID}" --out "outputs/story-demo-${SLURM_JOB_ID}"
