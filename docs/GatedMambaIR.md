# GatedMambaIR: Idea, Method & Experiments

_Last updated: 2026-09-27_

## Overview

GatedMambaIR is an image super-resolution model that replaces MambaIRv2's selective scan with a gated delta-style linear-attention scan. The ×2 model trained to 464k of its 500k iterations, and it trails MambaIRv2-Base on all five benchmarks by 0.25 to 1.49 dB PSNR. The ×3 and ×4 runs finished but were never tested, and their checkpoints are now missing. The largest gap is on Urban100. A code review points to a bug in the chunked scan, not to the idea itself.

## Idea and motivation

The idea is to give MambaIRv2's long-range branch a better memory by swapping the Mamba selective scan for a gated delta-rule scan, as in Gated DeltaNet.

- **Baseline.** MambaIRv2 combines window attention (for local detail) with an attentive state-space block. That block assigns each pixel to one of 128 learned semantic tokens and sorts pixels so similar ones sit next to each other. It then runs a one-directional Mamba scan over the sorted sequence.
- **Limitation targeted.** A Mamba scan keeps a compressed state that can only decay and add. It cannot overwrite a stale association, so the state fills up over long sequences such as a 200×200 test tile (40,000 tokens).
- **Proposal.** A gated delta scan keeps a matrix-valued memory `S` (value × key, per head). A decay gate α controls forgetting and a write gate β controls how strongly each token is stored. Before writing a new value, the delta rule also erases what `S` already predicts for that key. Repeated structure is then stored once instead of piling up.
- **Expected gain.** Better recall of repeated patterns that are far apart in the image, which should show most on Urban100 and Manga109.

## Methodology

GatedMambaIR keeps the full MambaIRv2 network and changes only the state-space block: `ASSM` becomes `GSSM` (`basicsr/archs/gatedmambair_arch.py`), and its selective scan becomes `GatedDeltaScan`.

**Network.** A 3×3 conv embeds the image to 174 channels, followed by six groups of six `AttentiveLayer`s and a pixel-shuffle upsampler. Each layer has two halves: shifted-window attention plus a ConvFFN, then GSSM plus a second ConvFFN.

**GSSM block.**

```
Input tokens ──► Conv 1x1 + CPE gate ──► Semantic sort ──► Gated scan ──► Norm, project, unsort
 (B, HW, 174)     (174 → 348)              ▲ index          ▲ prompt       (348 → 174)
      │                                    │                │
      └──────────► Router (pick 1 of 128) ─┴──► Prompt (token embedding, 16 → 348)
```

A router picks one of 128 semantic tokens for each pixel with a hard Gumbel-softmax. Pixels are sorted by that index so that similar content becomes neighbouring tokens. The scan runs over the sorted sequence and the result is unsorted afterwards. The chosen token's embedding becomes a prompt: it is projected from 16 to 348 channels and added to the q, k, v, α and β projections.

**Scan dimensions.** The hidden width is 348. `make_multihead_dims` produces 5 heads, each with 17 key channels and 69 value channels. q and k are L2-normalised. The decay gate is α = sigmoid(a) and the write gate is β = sigmoid(b).

**Target recurrence (Gated DeltaNet).**

$$
S_t = \alpha_t \, S_{t-1} \left(I - \beta_t k_t k_t^\top\right) + \beta_t v_t k_t^\top, \qquad o_t = S_t q_t
$$

**Implemented recurrence** (`basicsr/archs/chunk.py`, `chunk_gated_delta_fullrank`). The sequence is split into chunks of T = 256 tokens, and γ is the running product of α within a chunk. Each chunk reads only the state left by earlier chunks, then writes its own tokens:

$$
O_c = (\gamma \odot Q_c)\, S_{c-1}^\top, \qquad S_c = \gamma_T S_{c-1} + \sum_{t \in c} \frac{\gamma_T}{\gamma_t} \beta_t v_t k_t^\top
$$

This is gated linear attention with a β scale. It has no erase term and no read inside the chunk (see Analysis).

**Output.** The scan returns `x + r · W_o LN(o)`, where `r` is a learned scalar initialised to 0. q, k, v and the state are cast to bf16.

**Low-rank variant** (`use_lowrank`, off in all runs). It stores the state as S ≈ U Wᵀ with rank 16, and a gate g = G(k) drives both the U and W updates.

## Experimental setup

All runs use MambaIRv2-Base's classic-SR recipe, with only the scan changed. Each ran on one A100 80 GB (SLURM `pt` partition).

| Setting | Value |
| --- | --- |
| Training data | DF2K (DIV2K + Flickr2K), bicubic LR |
| Patch / batch | 128 px HR crop, batch 4, flips + rotations |
| Optimiser | Adam, lr 2.5e-5, betas (0.9, 0.99), no weight decay |
| Schedule | 500k iterations, lr halved at 250k, 400k, 450k, 475k |
| Loss | L1 |
| Network | embed 174, 6 groups × 6 layers, window 16, 128 semantic tokens, d_state 16, inner rank 64, MLP ratio 2 |
| Scan | 5 heads, key 17 / value 69 per head, chunk 256, full-rank state, prompt on |
| Parameters | 39.43M (×2), 39.61M (×3), 39.58M (×4) |
| Speed | 1.25 to 1.70 s per iteration |
| Validation | Set14 every 2k iterations |
| Test | Set5, Set14, B100, Urban100, Manga109; PSNR / SSIM on Y, border cropped by scale |
| Tiled inference | ~200 px LR tiles with 10% overlap (`GatedMambaIRModel.test`) |

Configs: `options/train/gatedmambair/`, `options/test/gatedmambair/`. The baseline is the official pretrained `experiments/pretrained_models/mambairv2_classicSR_Base_x2.pth`, run through the same test script. It was not retrained here.

## Results so far

At ×2, GatedMambaIR is 0.25 to 1.49 dB PSNR behind MambaIRv2-Base. The gap is largest on the two sets full of repeated structure.

| Test set | GatedMambaIR PSNR / SSIM | MambaIRv2-Base PSNR / SSIM | Δ PSNR (dB) |
| --- | --- | --- | --- |
| Set5 | 38.25 / 0.9615 | 38.65 / 0.9631 | −0.40 |
| Set14 | 33.93 / 0.9212 | 34.87 / 0.9275 | −0.94 |
| B100 | 32.37 / 0.9020 | 32.62 / 0.9052 | −0.25 |
| Urban100 | 33.00 / 0.9360 | 34.48 / 0.9468 | **−1.49** |
| Manga109 | 39.46 / 0.9788 | 40.42 / 0.9810 | −0.96 |

Sources: `logs/GatedMamabIRv2_inf_892727.out` (GatedMambaIR, 464k iterations) and `logs/MamabIRv2_inf_891729.out` (MambaIRv2-Base).

GatedMambaIR is also about 2.4× slower at test time. Urban100 took 31 min against 13 min, and Manga109 took 44 min against 18 min. The Python loop over chunks is the likely cause.

**Training runs**

| Run | Scale | Iterations | Best Set14 PSNR (val) | Status |
| --- | --- | --- | --- | --- |
| 820051 | ×2 | 464k of 500k | not in surviving logs | Crashed on resume (`expandable_segments` allocator error); tested at 464k |
| 819138 | ×3 | 500k (resumed from 356k) | 30.59 dB @ 476k | Finished 2026-01-05; not tested; checkpoints missing |
| 819139 | ×4 | 500k (resumed from 404k) | 28.80 dB @ 486k | Finished 2026-01-04; not tested; checkpoints missing |

The `models/` and `training_states/` folders for ×3 and ×4 are empty and were last modified on 2026-09-25. Logs from before the resumes are gone, so the early training curves are lost.

## Tracking experiments

Every run of `basicsr/test.py` appends one row per test set to `docs/results.csv`. A row holds the name, model, scale, dataset, PSNR, SSIM, image count, runtime, checkpoint iteration and path, config, git commit plus a dirty flag, and a tag. To label a run, set `val.tag` in the test YAML. To write to a different file, set `val.results_csv`.

```bash
# compare experiments (stdlib only, runs on a login node)
python basicsr/tools/compare_results.py --list
python basicsr/tools/compare_results.py --scale 2 --metric both --baseline MambaIRv2
python basicsr/tools/compare_results.py --scale 2 --baseline MambaIRv2 --markdown   # paste into this doc

# import old test logs written before the tracker existed
python basicsr/tools/import_test_log.py results/test_X/test_*.log --tag some-label
```

Experiments are keyed as `name@iter`. If the same checkpoint is tested twice, the newest row wins. For training curves (loss, Set14 validation PSNR), run `tensorboard --logdir tb_logger`. The event files there still hold the early training history that the SLURM logs lost.

## Analysis: why it trails MambaIRv2

The likely main cause is an implementation bug: each token can read only earlier chunks, never its own. Semantic sorting puts a token's most similar pixels in the same chunk, so exactly those are hidden. This matches the gap being largest on Urban100 and Manga109.

| # | Issue | Where | Effect |
| --- | --- | --- | --- |
| 1 | No within-chunk read: the output uses only the state `S` from earlier chunks | `chunk.py`, `chunk_gated_delta_fullrank` | The first 256 tokens output zero, and every token misses its nearest semantic neighbours |
| 2 | No delta erase term (`−β S k kᵀ`) | same | It is gated linear attention, not a delta rule, so the core idea is untested |
| 3 | Decay product underflows | `_compute_gamma` and the `γ_T / γ_t` scale | With α ≈ 0.5, 0.5²⁵⁶ rounds to 0 in fp32, so the write weights come out wrong |
| 4 | State kept in bf16 | `GatedDeltaScan.forward` | Rounding error builds up over ~160 chunks per 200 px test tile |
| 5 | Scan starts as an identity (`res_scale = 0`) and is followed by a second norm + projection | `GatedDeltaScan`, `GSSM` | Slower start, redundant layers |
| 6 | Updating U and W separately does not give the gated update of U Wᵀ | `chunk_gated_delta_lowrank` | Only matters if the variant is used |
| 7 | `key_ratio` / `head_dim` from the config are ignored (hardcoded 0.25 / 16) | `GSSM.__init__` | Configs cannot change head size |

Caveats on the comparison:

- The baseline uses official weights, while GatedMambaIR stopped at 464k of 500k iterations, before the last lr step at 475k.
- Git reports missing objects (`unable to read tree 56ecb38…`), so history on NFS scratch is not reliable.

## Scan fix (2026-09-27)

`GatedDeltaScan` now calls `chunk_gated_delta_rule` (`basicsr/archs/chunk.py`), which fixes issues 1–4 and 7:

- **Within-chunk read.** Each chunk's output = read from the state of earlier chunks + causal attention inside the chunk.
- **Log-space decay.** The decay comes from `logsigmoid(a)` and is only exponentiated as differences `g_i − g_j ≤ 0`, so there is no underflow or 0/0.
- **fp32.** q, k, v, the state and all accumulation stay in fp32; there is no bf16 anywhere in the scan.
- **Delta rule.** `use_delta: true` gives the real Gated DeltaNet update, with the within-chunk dependency solved by the WY representation (one unit-lower-triangular solve per chunk). `use_delta: false` gives gated linear attention, which is the ablation.
- **Chunk size.** `chunk_size` is now a network option (default 64); it was hardcoded to 256.
- **Config options.** `key_ratio` / `head_dim` from the config are passed through.

There are no new parameters, so old checkpoints still load. They were trained with the buggy scan, though, so they are at best a warm start. The low-rank variant (`use_lowrank`) is unchanged and still has issue 1.

Tests:

- `tests/test_gated_delta_scan.py` compares the chunked scan against a token-by-token fp32 recurrence (outputs, final state, gradients, strong decay, lengths that are not a multiple of the chunk). It also shows the old function's bug: the first chunk outputs exactly 0.
- `tests/test_gatedmambair_model.py` builds the full network with both variants, runs forward/backward on an odd-sized input, and loads an old checkpoint with `strict=True`.

## Next steps

1. **Back up the repo.** Copy the working tree and `.git` off scratch. If a copy of the ×3/×4 checkpoints exists anywhere, recover it.
2. **Write a reference test.** Compare the chunked scan against a naive token-by-token loop on random inputs, forward and backward. This proves issues 1 and 3 and guards every later fix.
3. **Fix the scan.** Add the within-chunk term, compute the decay in log space, and keep the state in fp32.
4. **Choose the rule.** Either keep gated linear attention and rename it, or add the delta erase using the chunked WY form. The `fla` library's `chunk_gated_delta_rule` kernel would also remove the Python loop.
5. **Run a short ablation before full training.** Train fixed-scan GatedMambaIR at ×2 for ~50k iterations and compare Set14 and Urban100 against MambaIRv2 trained the same way. Then run the full 500k ×2 training.
6. **Then train ×3 and ×4.** Fine-tune them from the ×2 weights, as MambaIRv2 does. Test all five sets and report parameters, FLOPs and runtime with `analysis/`.
7. **Housekeeping.** Pass `key_ratio` / `head_dim` through `GSSM` and drop the unused `self.act`. Once the allocator error is understood, remove the empty `PYTORCH_CUDA_ALLOC_CONF` workaround in `train.sh`.
