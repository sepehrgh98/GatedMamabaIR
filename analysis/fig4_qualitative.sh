#!/encs/bin/tcsh
#SBATCH --job-name=GMambaIR_fig4
#SBATCH --output=/speed-scratch/se_gham/GatedMambaIR/logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Fig. 4 qualitative candidates (x4): per-image metrics + shortlist + previews. CPU only.
# usage (from the repo root or a worktree): sbatch -p ps analysis/fig4_qualitative.sh [crops-only]
#   crops-only: reuse docs/fig4_qualitative/per_image_x4.csv, only redo shortlist / crops / previews
module purge
module load anaconda3/2023.03/default
conda activate /speed-scratch/se_gham/GatedMambaIR/GMambaIR_env
cd $SLURM_SUBMIT_DIR
echo "fig4 in `pwd` at `date`"
if ("$1" != "crops-only") then
  python analysis/per_image_metrics.py --root /speed-scratch/se_gham/GatedMambaIR --workers 8
endif
python analysis/fig4_qualitative.py --root /speed-scratch/se_gham/GatedMambaIR
echo "done! `date`"
