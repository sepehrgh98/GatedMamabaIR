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
