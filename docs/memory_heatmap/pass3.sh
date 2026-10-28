#!/encs/bin/tcsh
#SBATCH --job-name=GMambaIR_heatmap
#SBATCH --output=/speed-scratch/se_gham/GatedMambaIR/logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=pt
#SBATCH --gres=gpu:nvidia_a100_2g.20gb:1
#SBATCH --mem=40G
# pass 3: faster script (branch worktree-heatmap-speed: batched probes + causal cut), 256x256 crops on facades
module purge
module load gcc/9.3
module load anaconda3/2023.03/default
conda activate /speed-scratch/se_gham/GatedMambaIR/GMambaIR_env
setenv PYTORCH_CUDA_ALLOC_CONF ""
nvidia-smi -L
set LR = /speed-scratch/se_gham/GatedMambaIR/datasets/SR/Urban100/LR_bicubic/X2
set O = /speed-scratch/se_gham/GatedMambaIR/results/memory_heatmap/pass3
cd /nfs/speed-scratch/se_gham/GatedMambaIR/.claude/worktrees/heatmap-speed
echo "img012 start `date`"
python analysis/memory_heatmap.py --images $LR/img012x2.png --crop 100,256,256,256 --query 72,224 --query 52,218 --query 214,221 --layers 2.3 3.3 --out_dir $O/img012x2
echo "img073 start `date`"
python analysis/memory_heatmap.py --images $LR/img073x2.png --crop 120,256,256,256 --query 98,198 --query 108,214 --query 118,228 --layers 2.3 3.3 --out_dir $O/img073x2
echo "done! `date`"
