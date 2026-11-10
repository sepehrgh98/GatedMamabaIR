#!/encs/bin/tcsh
#SBATCH --job-name=GMambaIR_heatmap5
#SBATCH --output=/speed-scratch/se_gham/GatedMambaIR/logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=pt
#SBATCH --gres=gpu:nvidia_a100_2g.20gb:1
#SBATCH --mem=40G
# pass 5: a large batch for picking the Fig. 1 (motivation) examples.
# All 100 Urban100 x2 LR images, a 256x256 crop picked automatically where the structure repeats most,
# and 4 query points per crop picked automatically on repeating structure (analysis/heatmap_autopick.py).
# Same probe and display settings as pass 4 (the figure uses layer 2.3), so the maps are comparable.
# Resumable: images whose stats.csv already exists are skipped.
# At the end, sheet_L2.3.pdf (+ _index.txt) numbers every example for picking.
module purge
module load gcc/9.3
module load anaconda3/2023.03/default
conda activate /speed-scratch/se_gham/GatedMambaIR/GMambaIR_env
setenv PYTORCH_CUDA_ALLOC_CONF ""
nvidia-smi -L
set LR = /speed-scratch/se_gham/GatedMambaIR/datasets/SR/Urban100/LR_bicubic/X2
set O = /speed-scratch/se_gham/GatedMambaIR/results/memory_heatmap/pass5
cd /nfs/speed-scratch/se_gham/GatedMambaIR
foreach f ($LR/img*x2.png)
  set n = `basename $f .png`
  if (-e $O/$n/stats.csv) then
    echo "skip $n (done)"
    continue
  endif
  echo "$n start `date`"
  python analysis/memory_heatmap.py --images $f --crop auto --crop_size 256 --auto_queries 4 --min_sep 48 \
      --layers 2.3 --jac frob --n_probe 16 --route argmax --pct 98 --dilate 1 --out_dir $O/$n
end
python analysis/heatmap_sheet.py $O --layer 2.3
echo "done! `date`"
