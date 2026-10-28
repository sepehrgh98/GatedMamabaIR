# DeltaSR ablation design

_Written 2026-10-28. Results land in [results.csv](results.csv); job ids and progress are in [STATUS.md](STATUS.md)._

**DeltaSR** = lean GatedMambaIR with `gate: channel`, `use_delta: true`, `key_ratio: 0.25` / `head_dim: 87` (d_s = 87 keys, 1 head), `chunk_size: 128`, `use_prompt: true`. 25.1M params.

## Common setting

Every training run below is ×2, 50k iterations, the **same recipe as the existing 50k ablations**:

- DF2K, HR patch 128, batch 2 (one V100 32GB), seed 10.
- Adam lr 2e-4, MultiStepLR milestones [25k, 40k, 45k, 47.5k] × 0.5 (the 500k schedule ÷ 10).
- New configs are copies of `options/train/gatedmambair/train_GatedMambaIR_SR_x2_lean_chgate_50k.yml`, changing only the option(s) named in the row. The enlarged MambaIRv2 copies `options/train/mambairv2/train_MambaIRv2_SR_x2_50k.yml` and changes only `d_state`.
- Each run has a test config `options/test/.../test_<name>.yml`: Set5, Set14, B100, Urban100, Manga109, Y channel, crop_border 2, final checkpoint `net_g_50000.pth`. Scores go to `docs/results.csv` with tag `abl-<short name>`.

Existing runs are referenced by their `docs/results.csv` name (`test_<run>`); their configs are `options/{train,test}/.../<run>.yml`.

## Table 1: forgetting mechanism (replacement kept on)

Message: the memory must forget, and forgetting must be fine-grained and content-adaptive. One rate per head is not enough; no forgetting at all is worst.

| Row | Setting | Run | Status | Urban100 | Manga109 |
|---|---|---|---|---|---|
| ref | MambaIRv2 (22.9M) | `test_MambaIRv2_SR_x2_50k` | existing | 31.53 / 0.9230 | 38.38 / 0.9765 |
| exp1 | no forgetting (α = 1), `gate: none` | `GatedMambaIR_SR_x2_abl_noforget_50k` | **NEW** | | |
| exp2 | one rate per head, `gate: head` | `test_GatedMambaIR_SR_x2_lean_headgate_50k` | existing | 31.64 / 0.9240 | 38.44 / 0.9766 |
| ours | per-entry, content-adaptive, `gate: channel` | `test_GatedMambaIR_SR_x2_lean_chgate_50k` | existing | 31.70 / 0.9246 | 38.51 / 0.9768 |

## Table 2: replacement vs accumulation (2×2)

Message: what matters is forgetting; once the memory forgets per entry, replacing (delta rule) and accumulating (GLA) are equal. Without forgetting, replacement is what keeps the memory usable.

| Row | Forgetting | Write | Run | Status | Urban100 | Manga109 |
|---|---|---|---|---|---|---|
| exp1 | none | accumulate (`use_delta: false`) | `GatedMambaIR_SR_x2_abl_noforget_gla_50k` | **NEW** | | |
| exp2 | none | replace (`use_delta: true`) | same run as Table 1 exp1 | **NEW** | | |
| exp3 | per-entry | accumulate | `test_GatedMambaIR_SR_x2_lean_gla_50k` | existing | 31.74 / 0.9250 | 38.49 / 0.9768 |
| ours | per-entry | replace | `test_GatedMambaIR_SR_x2_lean_chgate_50k` | existing | 31.70 / 0.9246 | 38.51 / 0.9768 |

## Table 3: where the semantic prompt P enters

Message: the semantic prompt should steer *what is forgotten and how strongly it is written*, not only what is read. New option `prompt_gates` lists which of α (`a`) and β (`b`) also receive P; q always does.

| Row | P enters | `prompt_gates` | Run | Status | Urban100 | Manga109 |
|---|---|---|---|---|---|---|
| exp1 | q only | `[]` | `GatedMambaIR_SR_x2_abl_promptq_50k` | **NEW** | | |
| exp2 | q + α | `[a]` | `GatedMambaIR_SR_x2_abl_promptqa_50k` | **NEW** | | |
| exp3 | q + β | `[b]` | `GatedMambaIR_SR_x2_abl_promptqb_50k` | **NEW** | | |
| ours | q + α + β | `[a, b]` (default) | `test_GatedMambaIR_SR_x2_lean_chgate_50k` | existing | 31.70 / 0.9246 | 38.51 / 0.9768 |

## Table 4: capacity and efficiency

Message: giving MambaIRv2 the same parameter budget through a bigger state does not buy what DeltaSR's memory buys; DeltaSR's gain comes from how the memory is used, at a reasonable cost.

| Row | Model | Run | Status |
|---|---|---|---|
| 1 | MambaIRv2, d_state 16 (22.9M) | `test_MambaIRv2_SR_x2_50k` | existing |
| 2 | MambaIRv2, d_state enlarged to match 25.1M (value chosen in step 2 from built-model param counts) | `MambaIRv2_SR_x2_abl_dstateN_50k` | **NEW** |
| 3 | DeltaSR, d_s 30 (22.9M) | `test_GatedMambaIR_SR_x2_lean_k30_50k` | existing |
| 4 | DeltaSR, d_s 87 (25.1M) | `test_GatedMambaIR_SR_x2_lean_chgate_50k` | existing |

Columns: params, FLOPs, inference time and peak GPU memory on one 256×256 LR input (×2), Urban100 and Manga109 PSNR / SSIM. Params/FLOPs/time/memory come from `analysis/ablation_efficiency.py` (no training needed).

## Fig. A: memory size and chunk length (2 panels)

**(a) Urban100 PSNR vs d_s ∈ {16, 30, 48, 87}.** One line for replacement (`use_delta: true`), one for accumulation (`use_delta: false`); MambaIRv2 at both sizes (Table 4 rows 1-2) as reference points. d_s = `key_ratio` × 348 with `head_dim` = d_s (1 head): 16 → 0.046, 30 → 0.087, 48 → 0.138, 87 → 0.25 (checked from the built model).

Message: replacement uses a small memory better; the gap closes as the memory grows.

| d_s | replace (`use_delta: true`) | accumulate (`use_delta: false`) |
|---|---|---|
| 16 | **NEW** `GatedMambaIR_SR_x2_abl_ds16_50k` | **NEW** `GatedMambaIR_SR_x2_abl_ds16_gla_50k` |
| 30 | existing `test_GatedMambaIR_SR_x2_lean_k30_50k` (31.49) | **NEW** `GatedMambaIR_SR_x2_abl_ds30_gla_50k` |
| 48 | **NEW** `GatedMambaIR_SR_x2_abl_ds48_50k` | **NEW** `GatedMambaIR_SR_x2_abl_ds48_gla_50k` |
| 87 | existing `test_GatedMambaIR_SR_x2_lean_chgate_50k` (31.70) | existing `test_GatedMambaIR_SR_x2_lean_gla_50k` (31.74) |

**(b) Chunk length L ∈ {32, 64, 128, 160} vs inference time and peak GPU memory**, plus the recurrent (token-by-token) form as reference. Inference only, on the trained DeltaSR 50k weights; one 256×256 LR input. Also checks on one image that the outputs match across L (the chunk-wise form is exact). Message: chunking makes the exact recurrence fast; L trades time vs memory without changing the output.

## Fig. B: learned forgetting map (image)

For 2-3 Urban100 ×2 images with repeated facades and some sky: **input | per-pixel mean forget strength for ours | the same for Table 3 exp1 (P only in q)**. Forget strength = −log α = sigmoid(a) for the channel gate, averaged over the d_s memory entries. Middle layer (group 2 or 3, block 3), tokens un-sorted back to image positions as in `analysis/memory_heatmap.py`. Inference only (`analysis/forget_map.py`).

Message: with P in the gate, forgetting follows image content (e.g. strong at boundaries between regions, weak inside repeated structure); with P only in q, the map is flatter.

## Summary of new work

**11 new training runs** (each with a test job):

| # | Run | Changes vs `lean_chgate_50k` | Used in |
|---|---|---|---|
| 1 | `GatedMambaIR_SR_x2_abl_noforget_50k` | `gate: none` | Table 1 exp1, Table 2 exp2 |
| 2 | `GatedMambaIR_SR_x2_abl_noforget_gla_50k` | `gate: none`, `use_delta: false` | Table 2 exp1 |
| 3 | `GatedMambaIR_SR_x2_abl_promptq_50k` | `prompt_gates: []` | Table 3 exp1, Fig. B |
| 4 | `GatedMambaIR_SR_x2_abl_promptqa_50k` | `prompt_gates: [a]` | Table 3 exp2 |
| 5 | `GatedMambaIR_SR_x2_abl_promptqb_50k` | `prompt_gates: [b]` | Table 3 exp3 |
| 6 | `MambaIRv2_SR_x2_abl_dstateN_50k` | MambaIRv2 `d_state: N` | Table 4 row 2, Fig. A(a) |
| 7 | `GatedMambaIR_SR_x2_abl_ds16_50k` | `key_ratio: 0.046`, `head_dim: 16` | Fig. A(a) |
| 8 | `GatedMambaIR_SR_x2_abl_ds48_50k` | `key_ratio: 0.138`, `head_dim: 48` | Fig. A(a) |
| 9 | `GatedMambaIR_SR_x2_abl_ds16_gla_50k` | d_s 16, `use_delta: false` | Fig. A(a) |
| 10 | `GatedMambaIR_SR_x2_abl_ds30_gla_50k` | `key_ratio: 0.087`, `head_dim: 30`, `use_delta: false` | Fig. A(a) |
| 11 | `GatedMambaIR_SR_x2_abl_ds48_gla_50k` | d_s 48, `use_delta: false` | Fig. A(a) |

**Code options** (in `basicsr/archs/gatedmambair_arch.py`, all defaulting to current behaviour so existing configs and checkpoints are unchanged):

- `gate: none`: no forgetting (log α = 0); no `a_proj` / `ap_proj` parameters.
- `prompt_gates` (default `['a', 'b']`): which of α / β also receive P. `[]` = P only in q.
- A way to read the per-token log α of one layer in eval mode, for Fig. B.

**Analysis scripts** (no training): `analysis/ablation_efficiency.py` (Table 4 columns, Fig. A(b)), `analysis/forget_map.py` (Fig. B).
