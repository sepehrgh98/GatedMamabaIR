#!/encs/bin/tcsh
#SBATCH --job-name=GMambaIR_heatmap
#SBATCH --output=./logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=pt
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
# Motivation-figure heatmaps (see analysis/memory_heatmap.py). Edit the images / queries below, then:
#   sbatch analysis/memory_heatmap.sh
module purge
module load gcc/9.3
module load anaconda3/2023.03/default
conda activate /speed-scratch/se_gham/GatedMambaIR/GMambaIR_env
cd /speed-scratch/se_gham/GatedMambaIR
setenv PYTORCH_CUDA_ALLOC_CONF ""
nvidia-smi -L
echo "memory_heatmap at `date`"
set LR = datasets/SR/Urban100/LR_bicubic/X2
# 1) first pass without --query: saves <image>_grid.png (pick window pixels from it) and centre-query maps
python analysis/memory_heatmap.py --images $LR/img_004x2.png $LR/img_012x2.png $LR/img_073x2.png --layers 2.3 3.3
# 2) then, per image, rerun with chosen points, e.g.
# python analysis/memory_heatmap.py --images $LR/img_004x2.png --query 120,200 --query 60,310 --layers 2.3 3.3
echo "done! `date`"
