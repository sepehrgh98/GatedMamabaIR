#!/encs/bin/tcsh
#SBATCH --job-name=GMambaIR_fig4v3
#SBATCH --output=/speed-scratch/se_gham/GatedMambaIR/logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Fig. 4 search v3 (x4, all Urban100 + Manga109 images, no inference). CPU only.
# usage (from the repo root or a worktree): sbatch -p ps analysis/fig4_search_v3.sh
module purge
module load anaconda3/2023.03/default
conda activate /speed-scratch/se_gham/GatedMambaIR/GMambaIR_env
cd $SLURM_SUBMIT_DIR
echo "fig4 v3 in `pwd` at `date`"
python analysis/fig4_search_v3.py --root /speed-scratch/se_gham/GatedMambaIR --workers 8
echo "done! `date`"
