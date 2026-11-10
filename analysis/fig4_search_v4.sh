#!/encs/bin/tcsh
#SBATCH --job-name=GMambaIR_fig4v4
#SBATCH --output=/speed-scratch/se_gham/GatedMambaIR/logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Fig. 4 search v4 (x4, fine-pattern windows, all Urban100 + Manga109 images, no inference). CPU only.
# usage (from the repo root or a worktree): sbatch -p ps analysis/fig4_search_v4.sh
module purge
module load anaconda3/2023.03/default
conda activate /speed-scratch/se_gham/GatedMambaIR/GMambaIR_env
cd $SLURM_SUBMIT_DIR
echo "fig4 v4 in `pwd` at `date`"
python analysis/fig4_search_v4.py --root /speed-scratch/se_gham/GatedMambaIR --workers 8 --min_hp_std 5 --min_over_median 2.5 --min_gain 1.5 --min_score 0.05
echo "done! `date`"
