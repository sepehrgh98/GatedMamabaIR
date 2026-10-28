#!/encs/bin/tcsh
#SBATCH --job-name=GMambaIR_heatmap
#SBATCH --output=/speed-scratch/se_gham/GatedMambaIR/logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=pt
#SBATCH --gres=gpu:nvidia_a100_2g.20gb:1
#SBATCH --mem=40G
# pass 4, step 2: maps for 11 Urban100 x2 crops (256x256), 3 window queries each (crop coordinates).
# Queries were picked on *_grid.png from pass4_grids.sh and snapped to the darkest / brightest / bluest
# pixel within +-2 px; img012 / img073 (controls) reuse the pass-3 crops and queries.
module purge
module load gcc/9.3
module load anaconda3/2023.03/default
conda activate /speed-scratch/se_gham/GatedMambaIR/GMambaIR_env
setenv PYTORCH_CUDA_ALLOC_CONF ""
nvidia-smi -L
set LR = /speed-scratch/se_gham/GatedMambaIR/datasets/SR/Urban100/LR_bicubic/X2
set O = /speed-scratch/se_gham/GatedMambaIR/results/memory_heatmap/pass4
cd /nfs/speed-scratch/se_gham/GatedMambaIR/.claude/worktrees/heatmap-pass4
python analysis/memory_heatmap.py --images $LR/img001x2.png --crop 40,40,256,256 --query 95,169 --query 227,163 --query 177,79 --layers 2.3 3.3 --jac frob --n_probe 16 --route argmax --pct 98 --dilate 1 --out_dir $O/img001x2
python analysis/memory_heatmap.py --images $LR/img005x2.png --crop 100,40,256,256 --query 148,128 --query 209,93 --query 182,197 --layers 2.3 3.3 --jac frob --n_probe 16 --route argmax --pct 98 --dilate 1 --out_dir $O/img005x2
python analysis/memory_heatmap.py --images $LR/img015x2.png --crop 110,128,256,256 --query 237,118 --query 132,55 --query 217,235 --layers 2.3 3.3 --jac frob --n_probe 16 --route argmax --pct 98 --dilate 1 --out_dir $O/img015x2
python analysis/memory_heatmap.py --images $LR/img030x2.png --crop 120,40,256,256 --query 230,113 --query 176,194 --query 108,162 --layers 2.3 3.3 --jac frob --n_probe 16 --route argmax --pct 98 --dilate 1 --out_dir $O/img030x2
python analysis/memory_heatmap.py --images $LR/img034x2.png --crop 85,128,256,256 --query 210,181 --query 210,236 --query 154,211 --layers 2.3 3.3 --jac frob --n_probe 16 --route argmax --pct 98 --dilate 1 --out_dir $O/img034x2
python analysis/memory_heatmap.py --images $LR/img035x2.png --crop 60,40,256,256 --query 205,45 --query 108,151 --query 202,242 --layers 2.3 3.3 --jac frob --n_probe 16 --route argmax --pct 98 --dilate 1 --out_dir $O/img035x2
python analysis/memory_heatmap.py --images $LR/img041x2.png --crop 40,128,256,256 --query 148,207 --query 120,142 --query 178,249 --layers 2.3 3.3 --jac frob --n_probe 16 --route argmax --pct 98 --dilate 1 --out_dir $O/img041x2
python analysis/memory_heatmap.py --images $LR/img043x2.png --crop 40,128,256,256 --query 215,117 --query 153,186 --query 233,217 --layers 2.3 3.3 --jac frob --n_probe 16 --route argmax --pct 98 --dilate 1 --out_dir $O/img043x2
python analysis/memory_heatmap.py --images $LR/img067x2.png --crop 60,128,256,256 --query 202,103 --query 223,161 --query 148,152 --layers 2.3 3.3 --jac frob --n_probe 16 --route argmax --pct 98 --dilate 1 --out_dir $O/img067x2
python analysis/memory_heatmap.py --images $LR/img012x2.png --crop 100,256,256,256 --query 72,224 --query 52,218 --query 214,221 --layers 2.3 3.3 --jac frob --n_probe 16 --route argmax --pct 98 --dilate 1 --out_dir $O/img012x2
python analysis/memory_heatmap.py --images $LR/img073x2.png --crop 120,256,256,256 --query 98,198 --query 108,214 --query 118,228 --layers 2.3 3.3 --jac frob --n_probe 16 --route argmax --pct 98 --dilate 1 --out_dir $O/img073x2
echo "done! `date`"
