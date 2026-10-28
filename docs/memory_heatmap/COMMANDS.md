# Memory heatmap: selected examples

Script: analysis/memory_heatmap.py on branch worktree-heatmap-speed (commit ce4f873: batched probes +
causal cut at the query; same maps as the original, checked with --jac sum, max rel diff ~1e-7).
Models: defaults (MambaIRv2 x2 500k vs GatedMambaIR lean chgate x2 500k). --jac frob, 16 probes, argmax routing.
Run from the repo root (or the worktree), on a GPU node:

    LR=datasets/SR/Urban100/LR_bicubic/X2
    python analysis/memory_heatmap.py --images $LR/img012x2.png --crop 100,256,256,256 \
        --query 72,224 --query 52,218 --query 214,221 --layers 2.3 3.3 --out_dir results/memory_heatmap/pass3/img012x2
    python analysis/memory_heatmap.py --images $LR/img073x2.png --crop 120,256,256,256 \
        --query 98,198 --query 108,214 --query 118,228 --layers 2.3 3.3 --out_dir results/memory_heatmap/pass3/img073x2

(full sbatch launcher: pass3.sh; job 1529264, 2026-10-27, ~2 min per image on an A100 2g.20gb slice)
Query coordinates are in the 256x256 crop. Full-image img004 numbers (old script, pass 2):
stats_img004x2_fullimage.csv, command in results/memory_heatmap/pass2.sh on the cluster (not in git).

## Selected (clearest first)
1. img073x2_crop120_256_q108_214_L2.3 - glass tower pane. MambaIRv2: bright band on the rows just above the
   query (its sort-order neighbours). DeltaSR: points spread over the window grid of the same tower and the
   window rows of the building on the left.  eff_tokens 793 -> 615, top1% 0.78 -> 0.86, far 0.77 -> 0.97.
2. img012x2_crop100_256_q214_221_L3.3 - striped tower. MambaIRv2: few weak points near the query. DeltaSR:
   strong points along the window bands of the whole tower and the lit windows of the far tower (plus a faint
   haze over the sky).  eff_tokens 725 -> 6077, top1% 0.80 -> 0.44, far 0.91 -> 0.99.
3. img012x2_crop100_256_q72_224_L3.3 - orange building window. MambaIRv2 peak: roof edge of the other tower
   (unrelated). DeltaSR peaks: windows of the same shaded facade (plus faint haze).
   eff_tokens 538 -> 4538, top1% 0.82 -> 0.50, far 0.97 -> 0.98.
Note: in 2 and 3 the numbers say DeltaSR is MORE spread out (the faint haze counts), even though its
strongest points sit on the matching windows.

Files per example: <tag>.png (side by side), <tag>_input.png, <tag>_a.png (MambaIRv2), <tag>_b.png (DeltaSR).
