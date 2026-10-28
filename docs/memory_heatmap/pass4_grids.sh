#!/encs/bin/tcsh
#SBATCH --job-name=GMambaIR_heatmap
#SBATCH --output=/speed-scratch/se_gham/GatedMambaIR/logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=pt
#SBATCH --gres=gpu:nvidia_a100_2g.20gb:1
#SBATCH --mem=40G
# pass 4, step 1: (a) check that --pct/--dilate leave stats.csv unchanged vs pass 3 (img073),
# (b) grids (no --query) for the 256x256 facade crops, to pick query points.
module purge
module load gcc/9.3
module load anaconda3/2023.03/default
conda activate /speed-scratch/se_gham/GatedMambaIR/GMambaIR_env
setenv PYTORCH_CUDA_ALLOC_CONF ""
nvidia-smi -L
set LR = /speed-scratch/se_gham/GatedMambaIR/datasets/SR/Urban100/LR_bicubic/X2
set R = /speed-scratch/se_gham/GatedMambaIR/results/memory_heatmap
cd /nfs/speed-scratch/se_gham/GatedMambaIR/.claude/worktrees/heatmap-pass4
echo "check start `date`"
python analysis/memory_heatmap.py --images $LR/img073x2.png --crop 120,256,256,256 --query 98,198 --query 108,214 --query 118,228 --layers 2.3 3.3 --pct 98 --dilate 1 --out_dir $R/pass4_check/img073x2
cmp $R/pass3/img073x2/stats.csv $R/pass4_check/img073x2/stats.csv && echo "STATS IDENTICAL to pass 3"
foreach spec (001:40,40 005:100,40 015:110,128 030:120,40 034:85,128 035:60,40 041:40,128 043:40,128 067:60,128 012:100,256 073:120,256)
  set n = `echo $spec | cut -d: -f1`
  set rc = `echo $spec | cut -d: -f2`
  echo "grid img${n}x2 crop $rc start `date`"
  python analysis/memory_heatmap.py --images $LR/img${n}x2.png --crop $rc,256,256 --layers 2.3 3.3 --out_dir $R/pass4_grids/img${n}x2
end
echo "done! `date`"
