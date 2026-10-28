# Pass 4: exact commands

**Script:** `analysis/memory_heatmap.py` on branch `worktree-heatmap-pass4`.
- `3a2204a` adds the display-only `--dilate` option.
- The probing code is unchanged from pass 3 (`e362482`).

**Models:** the defaults, MambaIRv2 ×2 500k vs GatedMambaIR lean chgate ×2 500k.

**Runs:** one A100 2g.20gb slice, run from the repo root.

## Step 0: check that display options do not change the stats

```
python analysis/memory_heatmap.py --images $LR/img073x2.png --crop 120,256,256,256 \
    --query 98,198 --query 108,214 --query 118,228 --layers 2.3 3.3 --pct 98 --dilate 1 \
    --out_dir results/memory_heatmap/pass4_check/img073x2
```

The result was identical to pass 3's `stats.csv` except one value: `far_mass` 1.00000011 vs 1.0, which is float32 noise.

## Step 1: grids (no `--query`), used to pick query points

Launcher: `../pass4_grids.sh` (job 1530923). Here `LR=datasets/SR/Urban100/LR_bicubic/X2`.

```
python analysis/memory_heatmap.py --images $LR/img<NNN>x2.png --crop <r0>,<c0>,256,256 --layers 2.3 3.3 \
    --out_dir results/memory_heatmap/pass4_grids/img<NNN>x2
```

Crops (r0,c0): 001 40,40 · 005 100,40 · 015 110,128 · 030 120,40 · 034 85,128 · 035 60,40 · 041 40,128 ·
043 40,128 · 067 60,128 · 012 100,256 (control) · 073 120,256 (control).

## Step 2: maps

Launcher: `../pass4.sh` (job 1530963, 2026-10-27, ~2 min per image). Common options:
`--layers 2.3 3.3 --jac frob --n_probe 16 --route argmax --pct 98 --dilate 1`.

Query points (row,col) are in crop coordinates. Each was picked on a window that repeats in the crop, then snapped to the darkest, brightest or bluest pixel within ±2 px. The controls reuse the pass-3 points.

| Image | Crop | Queries |
|---|---|---|
| img001x2 | 40,40 | 95,169 · 227,163 · 177,79 |
| img005x2 | 100,40 | 148,128 · 209,93 · 182,197 |
| img015x2 | 110,128 | 237,118 · 132,55 · 217,235 |
| img030x2 | 120,40 | 230,113 · 176,194 · 108,162 |
| img034x2 | 85,128 | 210,181 · 210,236 · 154,211 |
| img035x2 | 60,40 | 205,45 · 108,151 · 202,242 |
| img041x2 | 40,128 | 148,207 · 120,142 · 178,249 |
| img043x2 | 40,128 | 215,117 · 153,186 · 233,217 |
| img067x2 | 60,128 | 202,103 · 223,161 · 148,152 |
| img012x2 (control) | 100,256 | 72,224 · 52,218 · 214,221 |
| img073x2 (control) | 120,256 | 98,198 · 108,214 · 118,228 |

Example (the others follow the same pattern; see `../pass4.sh`):

```
python analysis/memory_heatmap.py --images $LR/img035x2.png --crop 60,40,256,256 \
    --query 205,45 --query 108,151 --query 202,242 \
    --layers 2.3 3.3 --jac frob --n_probe 16 --route argmax --pct 98 --dilate 1 \
    --out_dir results/memory_heatmap/pass4/img035x2
```

All outputs (66 side-by-side PNGs) stay on the cluster in `results/memory_heatmap/pass4/`. Only the top 4 and all `stats.csv` files are in git.
