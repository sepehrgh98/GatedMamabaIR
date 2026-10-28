# Memory heatmaps (motivation figure)

For one query pixel on a window and one state-space layer: how strongly the scan output at that pixel
depends on every other pixel (`|d o_q / d x_i|`, Jacobian block norm). MambaIRv2 (`_a`) vs DeltaSR =
lean GatedMambaIR (`_b`), both ×2 500k, Urban100 LR ×2. Made with `analysis/memory_heatmap.py`;
exact commands in [COMMANDS.md](COMMANDS.md).

## Files

Per example `<tag>`:

- `<tag>.png`: side by side (input with query star | MambaIRv2 | DeltaSR)
- `<tag>_input.png`, `<tag>_a.png`, `<tag>_b.png`: single panels without axes, for composing the figure in LaTeX

`stats_*.csv` holds the concentration numbers for every query and layer, not only the selected ones.

## Selected examples (clearest first)

1. **`img073x2_crop120_256_q108_214_L2.3`** (glass tower pane), the best one.
   - MambaIRv2: a bright band on the rows just above the query (its neighbours in scan order).
   - DeltaSR: points spread over the window grid of the same tower and the windows of the building on the left.
2. **`img012x2_crop100_256_q214_221_L3.3`** (striped tower).
   - MambaIRv2: a few weak points near the query.
   - DeltaSR: strong points along the window bands of the whole tower and on the windows of a distant tower.
3. **`img012x2_crop100_256_q72_224_L3.3`** (orange building window).
   - MambaIRv2: its peak is an unrelated roof edge.
   - DeltaSR: its peaks are windows on the same facade.

## Honest reading, important for the paper

- **The pictures support the story:** DeltaSR's strongest points land on matching windows far away. MambaIRv2's land on its scan-order neighbours or on unrelated edges.
- **The numbers are mixed:**
  - **Layer 2.3:** DeltaSR is more concentrated (lower `eff_tokens`, higher `top1pct_mass`) at only 5 of 9 window points (img073 3/3, img004 2/3, img012 0/3).
  - **Layer 3.3:** DeltaSR is more spread out at all 9 of 9.
  - **`far_mass`:** DeltaSR's is higher or equal at 15 of 18.

  The reason is that DeltaSR also puts a faint, even haze over the whole image, sky included, and the concentration metrics count it.
- **So:** use the heatmaps as **qualitative** evidence ("DeltaSR reads from matching structure far away"). Do **not** claim the concentration numbers prove sharper focus.
- **Caveats:**
  - These are three hand-picked examples from 9 points.
  - img012 and img073 are 256×256 crops: the model only saw the crop.
  - The scan is causal over the sorted sequence, so a pixel can only read pixels sorted before it.

## Pass 4: larger pool (11 images, 33 query points, 66 pairs)

Files are in [`pass4/`](pass4/), with commands in [`pass4/COMMANDS.md`](pass4/COMMANDS.md).

**What was probed:**
- **Images:** 9 Urban100 facade crops (001, 005, 015, 030, 034, 035, 041, 043, 067) plus the controls 012 and 073, all 256×256 crops.
- **Queries:** 3 window points per image.
- **Layers:** 2.3 and 3.3.

**Display:** `--pct 98 --dilate 1`, so isolated dots show at print size. The stats are unchanged by this.

Top 4, ranked by eye. I penalised DeltaSR haze over flat regions.

1. **`img035x2_crop60_40_q205_45_L2.3`** (blue window grid)
   - MambaIRv2: a band of rows just above the query (recent scan neighbours) plus the mullion edges.
   - DeltaSR: the inside of every blue pane across the whole facade; the frames stay mostly empty.
2. **`img041x2_crop40_128_q178_249_L2.3`** (square windows)
   - MambaIRv2: only the squares along the query's own row.
   - DeltaSR: individual square windows all over the facade.
3. **`img005x2_crop100_40_q182_197_L2.3`** (curved curtain wall)
   - MambaIRv2: sparse dots on the white horizontal bars (unrelated edges).
   - DeltaSR: the dark glass panes across the facade, strongest above and left of the query.
4. **`img015x2_crop110_128_q217_235_L3.3`** (lit tower)
   - MambaIRv2: almost nothing, just scattered weak dots.
   - DeltaSR: the lit window rows across the facade. Weaker: its strongest weight sits in the ~60 rows above the query, with some haze.

**Concentration numbers do not favour DeltaSR for these four:**
- `eff_tokens` is 3–28× higher for DeltaSR.
- `top1pct_mass` is lower for DeltaSR.
- Only `far_mass` is higher or equal for DeltaSR, in 3 of 4.

Across all 66 pairs:
- DeltaSR has fewer `eff_tokens` in 9 pairs, all at layer 2.3 (9/33 there, 0/33 at layer 3.3).
- It has a higher `top1pct_mass` in 10 pairs.
- Its `far_mass` is higher or equal in 56 pairs.

The pictures and the numbers agree once you read them correctly: DeltaSR reads from *many* matching windows spread over the facade, which scores as "less concentrated". MambaIRv2 reads a narrow band of recent neighbours. So for the paper, describe the figure as "DeltaSR reads from repeated structure across the image", not "DeltaSR is more focused".
