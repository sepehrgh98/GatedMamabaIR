#!/encs/bin/tcsh
#SBATCH --job-name=GMambaIR_analysis
#SBATCH --output=./logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=pt
#SBATCH --mem=40G
# usage: sbatch --gres=gpu:1 --constraint=Model-V100 analysis/gpu_job.sh <python script> [args...]
#   e.g. sbatch --gres=gpu:1 --constraint=Model-V100 analysis/gpu_job.sh analysis/ablation_efficiency.py
module purge
module load gcc/9.3
module load anaconda3/2023.03/default
conda activate /speed-scratch/se_gham/GatedMambaIR/GMambaIR_env
cd /speed-scratch/se_gham/GatedMambaIR
setenv PYTORCH_CUDA_ALLOC_CONF ""
setenv ATTN_MODE "xformers"
nvidia-smi -L
echo "Running $argv at `date`"
python $argv
echo "done! `date`"
