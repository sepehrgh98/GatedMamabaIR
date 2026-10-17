#!/bin/bash
#SBATCH --job-name=GMambaIR_genLR
#SBATCH --output=./logs/%x_%j.out
#SBATCH --partition=ps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=04:00:00
# Regenerate DF2K bicubic LR (X2/X3/X4) from HR after the scratch purge. Submit from repo root.
cd /speed-scratch/se_gham/GatedMambaIR
/speed-scratch/se_gham/GatedMambaIR/GMambaIR_env/bin/python basicsr/tools/generate_bicubic_lr.py \
  --hr datasets/DF2K/DF2K/HR --out datasets/DF2K/DF2K/LR_bicubic --scales 2 3 4 --workers $SLURM_CPUS_PER_TASK
for s in 2 3 4; do echo "X$s: $(ls datasets/DF2K/DF2K/LR_bicubic/X$s | wc -l) files"; done
test $(ls datasets/DF2K/DF2K/LR_bicubic/X2 | wc -l) -eq 3550
