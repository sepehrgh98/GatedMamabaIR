#!/encs/bin/tcsh
#SBATCH --job-name=GMambaIR_test
#SBATCH --output=./logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=pt
#SBATCH --mem=40G
module purge
module load gcc/9.3
module load anaconda3/2023.03/default
conda activate /speed-scratch/se_gham/GatedMambaIR/GMambaIR_env
cd /speed-scratch/se_gham/GatedMambaIR
setenv PYTORCH_CUDA_ALLOC_CONF ""
setenv ATTN_MODE "xformers"
nvidia-smi -L
echo "Testing $1 at `date`"
python basicsr/test.py -opt $1
echo "done! `date`"
