# DeltaSR ablation design

_Written 2026-10-28; results added 2026-10-31. Results land in [results.csv](results.csv); job ids and progress are in [STATUS.md](STATUS.md)._

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
| 2 | MambaIRv2, **d_state 72** (25,136,729 params vs DeltaSR 25,124,201, +0.05%) | `MambaIRv2_SR_x2_abl_dstate72_50k` | **NEW** |

d_state choice: MambaIRv2 grows by exactly 39,888 params per unit of d_state (22,903,001 at 16). 72 is the closest match to DeltaSR (71 gives 25,096,841, −0.11%).
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

| # | Run | Changes vs `lean_chgate_50k` | Params | Used in |
|---|---|---|---|---|
| 1 | `GatedMambaIR_SR_x2_abl_noforget_50k` | `gate: none` | 23,981,021 | Table 1 exp1, Table 2 exp2 |
| 2 | `GatedMambaIR_SR_x2_abl_noforget_gla_50k` | `gate: none`, `use_delta: false` | 23,981,021 | Table 2 exp1 |
| 3 | `GatedMambaIR_SR_x2_abl_promptq_50k` | `prompt_gates: []` | 25,073,513 | Table 3 exp1, Fig. B |
| 4 | `GatedMambaIR_SR_x2_abl_promptqa_50k` | `prompt_gates: [a]` | 25,123,625 | Table 3 exp2 |
| 5 | `GatedMambaIR_SR_x2_abl_promptqb_50k` | `prompt_gates: [b]` | 25,074,089 | Table 3 exp3 |
| 6 | `MambaIRv2_SR_x2_abl_dstate72_50k` | MambaIRv2 `d_state: 72` | 25,136,729 | Table 4 row 2, Fig. A(a) |
| 7 | `GatedMambaIR_SR_x2_abl_ds16_50k` | `key_ratio: 0.046`, `head_dim: 16` | 22,371,389 | Fig. A(a) |
| 8 | `GatedMambaIR_SR_x2_abl_ds48_50k` | `key_ratio: 0.138`, `head_dim: 48` | 23,612,093 | Fig. A(a) |
| 9 | `GatedMambaIR_SR_x2_abl_ds16_gla_50k` | d_s 16, `use_delta: false` | 22,371,389 | Fig. A(a) |
| 10 | `GatedMambaIR_SR_x2_abl_ds30_gla_50k` | `key_ratio: 0.087`, `head_dim: 30`, `use_delta: false` | 22,914,197 | Fig. A(a) |
| 11 | `GatedMambaIR_SR_x2_abl_ds48_gla_50k` | d_s 48, `use_delta: false` | 23,612,093 | Fig. A(a) |

Reference sizes: DeltaSR (d_s 87) 25,124,201; DeltaSR d_s 30 22,914,197; MambaIRv2 (d_state 16) 22,903,001. Key counts checked from the built models (1 head; 16 / 30 / 48 / 87 keys). `gate: none` drops `a_proj` / `ap_proj` (−1.14M). The write rule (`use_delta`) does not change the size.

**Code options** (in `basicsr/archs/gatedmambair_arch.py`, all defaulting to current behaviour so existing configs and checkpoints are unchanged):

- `gate: none`: no forgetting (log α = 0); no `a_proj` / `ap_proj` parameters.
- `prompt_gates` (default `['a', 'b']`): which of α / β also receive P. `[]` = P only in q.
- `record_log_alpha` on a `GatedDeltaScan` (eval only): keeps that layer's log α in `last_log_alpha`, un-sorted by GSSM to image order as `[B, H, W, n_gate]`, for Fig. B.

Tests: `tests/test_ablation_options.py` (existing 50k checkpoints give bit-identical outputs before/after the change; variants build, train and round-trip; the recorded map is in image order), `tests/test_gated_delta_scan.py` (adds a zero-decay case), `tests/check_ablation_variants.py` (GPU: a few training steps of every new config at the real size, finite gradients, peak memory).

**Analysis scripts** (no training): `analysis/ablation_efficiency.py` (Table 4 columns, Fig. A(b)), `analysis/forget_map.py` (Fig. B). Launcher: `sbatch --gres=gpu:1 --constraint=Model-V100 analysis/gpu_job.sh <script> [args]`.

## Results (as of 2026-10-31 evening)

PSNR (dB) / SSIM on the Y channel, ×2, 50k iterations, same recipe for every row. Single runs: treat differences under ~0.05 dB as noise. **Bold** = best in the table. Raw numbers: [results.csv](results.csv) (tags `abl-*` for the new runs).

**Status:** Tables 1-4, Fig. A(b) and Fig. B are complete. Fig. A(a) has 4 of 8 points; the rest arrive 2026-10-31 night (accumulate @ 30) and 2026-11-01 evening (replace @ 16, accumulate @ 48).

### Table 1: forgetting mechanism (replacement kept on) — complete

| Row | Set5 | Set14 | B100 | Urban100 | Manga109 |
|---|---|---|---|---|---|
| MambaIRv2 (ref) | 37.91 / 0.9602 | 33.42 / 0.9169 | 32.08 / 0.8985 | 31.53 / 0.9230 | 38.38 / 0.9765 |
| exp1: no forgetting | 37.90 / 0.9604 | 33.44 / 0.9171 | 32.09 / 0.8988 | 31.62 / 0.9240 | 38.45 / 0.9766 |
| exp2: one rate per head | 37.91 / 0.9603 | 33.41 / 0.9168 | 32.10 / 0.8989 | 31.64 / 0.9240 | 38.44 / 0.9766 |
| **ours: per entry, content-adaptive** | **37.95 / 0.9604** | **33.46 / 0.9172** | **32.11 / 0.8990** | **31.70 / 0.9246** | **38.51 / 0.9768** |

Reading: ordered as expected (none < per head < per entry), but the steps are small (+0.08 dB Urban100 from no forgetting to ours): with replacement on, the memory is already usable without forgetting.

### Table 2: replacement vs accumulation (2×2) — complete

| Forgetting | Write | Set5 | Set14 | B100 | Urban100 | Manga109 |
|---|---|---|---|---|---|---|
| none | accumulate | 37.83 / 0.9601 | 33.31 / 0.9160 | 32.02 / 0.8979 | **31.29** / 0.9206 | 38.19 / 0.9759 |
| none | replace | 37.90 / 0.9604 | 33.44 / 0.9171 | 32.09 / 0.8988 | 31.62 / 0.9240 | 38.45 / 0.9766 |
| per entry | accumulate | 37.94 / 0.9605 | 33.46 / 0.9172 | 32.11 / 0.8990 | **31.74 / 0.9250** | 38.49 / 0.9768 |
| per entry (ours) | replace | **37.95** / 0.9604 | 33.46 / 0.9172 | 32.11 / 0.8990 | 31.70 / 0.9246 | **38.51 / 0.9768** |

Reading (clearest result): with neither forgetting nor replacement the memory collapses (Urban100 31.29, below MambaIRv2's 31.53). Either mechanism alone fixes it (31.62 / 31.74), and with per-entry forgetting the two write rules tie.

### Table 3: where the semantic prompt P enters — complete

| P enters | Set5 | Set14 | B100 | Urban100 | Manga109 |
|---|---|---|---|---|---|
| exp1: q only | 37.93 / 0.9604 | 33.42 / 0.9172 | 32.10 / 0.8989 | 31.64 / 0.9243 | 38.47 / 0.9767 |
| exp2: q + α | 37.86 / 0.9602 | 33.38 / 0.9165 | 32.07 / 0.8986 | 31.48 / 0.9226 | 38.39 / 0.9765 |
| exp3: q + β | 37.85 / 0.9601 | 33.36 / 0.9162 | 32.06 / 0.8982 | 31.43 / 0.9218 | 38.36 / 0.9764 |
| **ours: q + α + β** | **37.95 / 0.9604** | **33.46 / 0.9172** | **32.11 / 0.8990** | **31.70 / 0.9246** | **38.51 / 0.9768** |

Reading: the full design is best, but adding P to only one gate is *worse* than q only (−0.16 / −0.21 dB Urban100). P helps only when it steers forgetting and writing together. The dip is large for single runs; a second seed would make it firmer.

### Table 4: capacity and efficiency — complete

V100, one 256×256 LR input, ×2 ([ablation_efficiency/](ablation_efficiency/)).

| Model | Params | GFLOPs | Time | Peak memory | Urban100 | Manga109 |
|---|---|---|---|---|---|---|
| MambaIRv2, d_state 16 | 22.90M | 1,784 | 1.99 s | 1,675 MiB | 31.53 / 0.9230 | 38.38 / 0.9765 |
| MambaIRv2, d_state 72 | 25.14M | 2,306 | 2.32 s | 1,687 MiB | **31.82 / 0.9258** | **38.56 / 0.9769** |
| DeltaSR, d_s 30 | 22.91M | 1,876 | 4.39 s | 1,693 MiB | 31.49 / 0.9226 | 38.39 / 0.9764 |
| DeltaSR, d_s 87 (ours) | 25.12M | 2,196 | 4.87 s | 1,860 MiB | 31.70 / 0.9246 | 38.51 / 0.9768 |

Reading: **the intended message does not hold at 50k.** At the same parameter budget, MambaIRv2 with a larger state beats DeltaSR on all five sets (+0.12 Urban100, +0.05 Manga109; Set5 38.00 / Set14 33.49 / B100 32.13). DeltaSR also runs ~2.1× slower at fewer FLOPs (Python chunk loop + per-chunk triangular solve vs MambaIRv2's fused CUDA scan). Open: a 500k run of MambaIRv2 d_state 72 (our 500k win was against d_state 16), or reframing this table.

### Fig. A(a): Urban100 PSNR vs memory size d_s — 7 of 8 points (1 provisional)

| d_s | replace (`use_delta: true`) | accumulate (`use_delta: false`) |
|---|---|---|
| 16 | pending (test running 2026-11-01) | 31.65 / 0.9242 |
| 30 | 31.49 / 0.9226 | 31.50 / 0.9228 |
| 48 | 31.51 / 0.9228 | 30.82 / 0.9151 **(provisional, see below)** |
| 87 | 31.70 / 0.9246 | 31.74 / 0.9250 |

Reference points: MambaIRv2 d_state 16 = 31.53, d_state 72 = 31.82.

**Provisional point (accumulate @ 48, seed 10):** this run fell behind at ~15-20k (Set14 validation 31.38 at 20k vs 32.5-32.8 for the other runs) and never caught up (33.12 at 50k vs 33.4-33.5); it is lower on all five test sets. Config, checkpoint and data were checked; no NaNs, no loss spikes, no restart. Most likely an unlucky single run (batch 2 makes 50k training noisy; it also ran on an A100 slice, where convolutions use TF32). A re-run with seed 20 on a V100 (train 1555864, test 1555865, config `*_abl_ds48_gla_seed20_50k.yml`) will **replace** this point. Do not use 30.82 in the paper unmarked.

Reading so far: replacement improves with memory size (31.49 → 31.51 → 31.70). Accumulation is flat at 16 / 30 (31.65 / 31.50) and best at 87 (31.74); the two write rules tie at 30 and 87. The "replacement uses a small memory better" hypothesis is not supported. At 50k, single-run noise (~0.1-0.3 dB) is as large as most differences here; the clear point is that d_s 87 is best for both rules.

### Fig. A(b): chunk length — complete

DeltaSR d_s 87, trained 50k weights, V100, one 256×256 LR input ([chunks.png](ablation_efficiency/chunks.png)).

| Form | Time | Peak memory | Max output diff vs L = 128 |
|---|---|---|---|
| chunk L = 32 | 12.78 s | 1,808 MiB | 0.00006 |
| chunk L = 64 | 7.08 s | 1,830 MiB | 0.0029 |
| **chunk L = 128 (used)** | **4.87 s** | 1,866 MiB | 0 |
| chunk L = 160 | 4.15 s | 2,040 MiB | 0.00006 |
| recurrent (token by token) | 437.2 s | 1,699 MiB | 0.0026 |

Output differences are on the [0, 1] image scale (0.003 ≈ 0.7 of one gray level). Chunking is ~90× faster than the recurrence; longer chunks are faster but use more memory. The two larger differences (L = 64, recurrent) are bigger than float round-off; a likely cause is the order of tied tokens in the unstable semantic sort changing between passes, not the chunking itself (not yet verified).

### Fig. B: learned forgetting map — complete

[forget_map/](forget_map/): Urban100 img014 / img034 / img087, GSSM layer 2.3, input | ours | P only in q, shared colour scale.

| Image | ours: mean forget strength | P only in q |
|---|---|---|
| img014 | 0.727 | 0.774 |
| img034 | 0.723 | 0.772 |
| img087 | 0.729 | 0.777 |

Reading: both maps follow content (sky flat, facade follows windows and edges), since the gate also sees the pixel features. With P in the gate, the model forgets less overall and holds memories longer inside the repeated facade; with P only in q it forgets more everywhere, with a noisier pattern. (The design's guess "flatter without P" is not what we see.)
