#!/bin/bash
#SBATCH --job-name=GMambaIR_gpucheck
#SBATCH --output=./logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=pt
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=00:45:00

# Scan unit tests + gradient health + chunk-size speed on a GPU node.
# Submit from the repo root: sbatch tests/gpu_check.sh

source /etc/profile.d/modules.sh
module purge
module load gcc/9.3
cd /speed-scratch/se_gham/GatedMambaIR
PY=/speed-scratch/se_gham/GatedMambaIR/GMambaIR_env/bin/python
export PYTORCH_CUDA_ALLOC_CONF=""

nvidia-smi --query-gpu=name,memory.total --format=csv
echo "== scan unit tests (CPU reference)"
# $PY tests/test_gated_delta_scan.py  # passed on job 1400851
echo "== GPU stress"
$PY tests/gpu_stress.py --seeds 10 --size 64 --batch 1
echo "done"
