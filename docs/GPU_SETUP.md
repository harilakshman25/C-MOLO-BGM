# V100 setup after successful allocation probe

User-supplied job 29910 on 25 October 2026 confirmed gnode1, one allocated GPU
(CUDA_VISIBLE_DEVICES=1), V100-PCIE-32GB hardware and driver 580.95.05.
Two inventory rows do not mean two GPUs were allocated. Shared project path:
/home/bt3/23CS10009/C-MOLO-BGM.

## Install on gpu73, without using its GPUs

After pushing local changes to GitHub, run in the gpu73 VS Code terminal:

```bash
cd ~/C-MOLO-BGM
git pull --ff-only
conda create -n cmolo-gpu python=3.11 pip -y
conda activate cmolo-gpu
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-gpu-v100.txt
python -m pip install -e .
python -m pip check
mkdir -p logs
python -m pip freeze > logs/cmolo-gpu-installed.txt
```

If the environment already exists, activate it instead of repeating conda create.
Package installation needs network access and several GB of disk, but does not launch
GPU inference. Keep cmolo-cpu unchanged. Stop and share the error if an install fails.
Do not install the broad inference extra over this deliberately pinned candidate stack.

We use PyTorch 2.7.1's official CUDA 11.8 wheel as a conservative V100 candidate, not the
newest release. Recent CUDA 12.8+ builds have removed Volta support. The driver does not
need to match the wheel's CUDA runtime number exactly. No system CUDA toolkit or driver
installation is requested. FP16 is the initial precision; do not assume native BF16 on V100.

## Test inside a Slurm allocation

In a fresh SSH terminal on cluster100:

```bash
cd ~/C-MOLO-BGM
conda activate cmolo-gpu
mkdir -p logs
sbatch --partition=gpupart_v100 jobs/torch_check.sbatch "$CONDA_PREFIX/bin/python"
squeue -u "$USER"
```

The environment must be visible from the shared home on the compute node. An absolute
Python path avoids needing conda initialization in a batch shell. Do not set GPU visibility
manually. Use the returned job ID below:

```bash
cat logs/cmolo-torch-JOBID.out
cat logs/cmolo-torch-JOBID.err
```

Expected: GPU CHECK PASSED, V100 name, capability [7,0], successful FP16 matrix operation,
and StableAudioPipeline import. This establishes actual PyTorch GPU execution rather than
just nvidia-smi inventory. No model weights are downloaded by this test.

The setup and GPU test are prepared but not executed on the server by the assistant.
After success, the next step is gated-model access/download on gpu73, a single 5-second
baseline generation job, and only then training integration. Model license/access requires
the user's own Hugging Face account; never put tokens in repository files or shared logs.

Sources: [official PyTorch install commands](https://pytorch.org/get-started/previous-versions/),
[Volta support changes](https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/3290).
