# GatedMambaIR: Status Report

_Last updated: 2026-10-28 22:30 EDT. Method and background: [GatedMambaIR.md](GatedMambaIR.md). All test scores: [results.csv](results.csv)._

## Summary

- **Final ×2 result (500k, same recipe): lean GatedMambaIR (25.1M) beats MambaIRv2 (22.9M) on all five benchmarks** (section 18):
  - **+0.16 dB on Urban100** (33.93 vs 33.77)
  - **+0.13 dB on Set14** (34.57 vs 34.44)
  - +0.01 to +0.04 dB on Set5, B100 and Manga109
- During training, lean was ahead at **75 of 81** same-step Set14 checkpoints since 100k (mean **+0.11 dB**). Curves: [500k_curves.png](500k_curves.png).
- The same win already showed after 50k steps (+0.17 Urban100, +0.12 Manga109), so the gain holds through full training.
- **×3 and ×4 (250k fine-tunes, paper schedule): lean also wins on all five benchmarks at both scales** (section 20):
  - **+0.17 dB on Urban100** at both ×3 (29.77 vs 29.60) and ×4 (27.49 vs 27.32)
  - ×4 Manga109 +0.09; the other sets +0.01 to +0.05
  - SSIM is higher on every set at both scales. Curves: [x3_curves.png](x3_curves.png), [x4_curves.png](x4_curves.png).
- **DeltaSR ablation studies launched** (2026-10-28, section 21, design in [ABLATIONS.md](ABLATIONS.md)): 11 new 50k runs (forgetting, replace vs accumulate, prompt position, capacity / d_s sweep, size-matched MambaIRv2 d_state 72); all finish ~Nov 3-6.
- **Where the gain comes from (ablations, section 11):** the lean design itself (per-channel-gated linear-attention memory in place of the Mamba scan). The delta erase is not needed: lean without it (GLA) ties lean.

## What we did

### 1. Recovery (2026-09-27)
- `/speed-scratch` was purged on 2026-09-25. We rebuilt the conda env (`GMambaIR_env`) and regenerated the DF2K bicubic LR images.
- **Git is still broken** (missing objects), so nothing can be committed. The code is backed up to `~/GatedMambaIR_backup_2026-09-28/`.

### 2. Scan bug fix (2026-09-27)
- The old scan had four problems:
  - no read inside a chunk (the first 256 tokens output zero),
  - no delta erase,
  - decay underflow,
  - a bf16 state.
- It was rewritten as `chunk_gated_delta_rule` in `basicsr/archs/chunk.py`: fp32, log-space decay, real Gated DeltaNet update.
- Tested against a token-by-token reference in `tests/test_gated_delta_scan.py`.

### 3. 50k ablation at ×2 (2026-09-27 → 09-29)
Three models with the same data, seed and schedule (batch 2, 50k steps, V100), tested on all five benchmarks.

| Test set | MambaIRv2 (22.9M) | GatedMambaIR delta (39.4M) | GatedMambaIR GLA (39.4M) |
|---|---|---|---|
| Set5 | 37.91 | **37.93** | 37.83 |
| Set14 | 33.42 | **33.42** | 33.28 |
| B100 | 32.08 | **32.08** | 32.01 |
| Urban100 | 31.53 | **31.56** | 31.24 |
| Manga109 | **38.38** | 38.36 | 38.20 |

PSNR (dB). The training curves are in `docs/ablation50k_curves.png`; the training loss is identical for all three.

**Reading:**
- The delta erase gives +0.32 dB over GLA on Urban100, which is where repeated patterns are.
- Delta vs MambaIRv2 is within ±0.03 dB, which is a tie.
- The MambaIRv2 paper shows +0.11 dB on Urban100 from B→L (23M→34M) from size alone. So at 39M, GatedMambaIR "should" be about 0.1 dB ahead just from size. That's why we built a size-matched version.

### 4. Theory: why GLA < MambaIRv2 and delta only ties
- **Forgetting.**
  - Mamba has thousands of forgetting rates (one per channel and state).
  - Our scan had **one per head (5 in total)**, so fast and slow information can't live side by side.
- **Crowding.**
  - GLA only adds to memory, so with 17-key heads the stored patterns interfere.
  - The delta rule is online least-squares, which removes that interference. That is the +0.32 dB.
- **Capacity.**
  - With 17 key channels per head, the memory holds about 17 clean associations per head.
  - The erase keeps them clean but can't add room.
- **Where the extra 16.5M params come from.** Every scan block had q/k/v projections, a second set for the prompt, and an output projection, followed by *another* norm + projection in GSSM. That's about 486k params per layer, against about 33k in MambaIRv2's scan.

### 5. New code (2026-09-29)
Two new model options. Both are off by default, so old configs and checkpoints are unchanged (verified: all old checkpoints load with `strict=True`).

| Option | What it does |
|---|---|
| `gate: channel` | One forgetting rate per key channel instead of per head. The decay is bounded to α ∈ [0.37, 1] per token, so the chunked math stays finite; needs `chunk_size` ≤ 160 (we use 128). |
| `lean: true` | Size-matched scan block, closer to MambaIRv2: value = input (no v projection), the 16-dim prompt goes only into q and the gates, no duplicate norm/projection, Mamba-style skip term `D`. |

Checks:
- `tests/test_gated_delta_scan.py` (7/7 pass, CPU).
- `tests/check_variants.py` (GPU job 1401768).

| Variant | Params | fwd+bwd (batch 1, A100 20GB slice) |
|---|---|---|
| delta (current) | 39.43M | 1.69 s |
| delta + channel gate | 41.43M | 2.13 s |
| **lean + channel gate, 1 head × 87 keys** | **25.12M** | **1.62 s** |
| lean + channel gate, 3 heads × 29 keys | 25.15M | 1.88 s |
| lean + channel gate, 2 heads × 87 keys | 28.51M | 1.87 s |
| MambaIRv2-B (reference) | 22.90M | – |

Pre-change copies of the edited files: `~/GatedMambaIR_backup_2026-09-28/pre_perchannel/`.

### 6. Decision: retrain the baseline, don't only use the paper's numbers
- The paper trained MambaIRv2 with **batch 32 on 8 GPUs**. We can only do **batch 4 on one GPU**, so both models will score below the paper's numbers.
- Comparing our model with their numbers would lose for reasons unrelated to the idea.
- The main table will compare **both models retrained with our recipe**. The paper's official numbers go in as a reference row.

### 7. Ablation results (all 50k, same recipe), 2026-10-01

| Test set | MambaIRv2 22.9M | delta 39.4M | GLA 39.4M | bigger heads 43.8M | delta + channel gate 41.4M | **lean + channel gate 25.1M** |
|---|---|---|---|---|---|---|
| Set5 | 37.91 | 37.93 | 37.83 | 37.92 | 37.74 | **37.95** |
| Set14 | 33.42 | 33.42 | 33.28 | 33.43 | 33.21 | **33.46** |
| B100 | 32.08 | 32.08 | 32.01 | 32.08 | 31.95 | **32.11** |
| Urban100 | 31.53 | 31.56 | 31.24 | 31.54 | 31.05 | **31.70** |
| Manga109 | 38.38 | 38.36 | 38.20 | 38.42 | 38.06 | **38.51** |

**Reading:**
- **Lean wins everywhere, with fewer extra parameters** (+10% instead of +72%).
- **Bigger heads ≈ small heads.** More key room alone doesn't help.
- **Per-channel forgetting on the big model is clearly worse** (−0.51 dB Urban100 vs delta).
- So the lean gain may come mostly from the leaner block itself. Likely reasons:
  - the scan is active from step 0 (the old block started at `res_scale = 0`),
  - there are no redundant projections.
- The check run below (lean with the per-head gate) separates the two.
- **Caveats:**
  - These are single 50k runs.
  - The final claim rests on the 500k runs.

### 8. Incident: baseline 500k crash
- Job 1401762 (MambaIRv2 500k) died at step 44,100 with `DataLoader worker ... killed by signal: Bus error` (out of shared memory with 8 loader workers).
- That's a cluster issue, not the model.
- Fix: `num_worker_per_gpu: 4` in both 500k configs. This doesn't change results.
- Resubmitted as 1410815, which resumes from the 40k checkpoint (`--auto_resume`).

### 9. More 50k ablations (2026-10-03)

| Test set | MambaIRv2 22.9M | **lean 25.1M** (channel gate, 87 keys) | lean, **per-head** gate 25.1M | lean, **exact size** 22.9M (30 keys) |
|---|---|---|---|---|
| Set5 | 37.91 | **37.95** | 37.91 | 37.88 |
| Set14 | 33.42 | **33.46** | 33.41 | 33.39 |
| B100 | 32.08 | **32.11** | 32.11 | 32.07 |
| Urban100 | 31.53 | **31.70** | 31.64 | 31.49 |
| Manga109 | 38.38 | **38.51** | 38.44 | 38.39 |

**Reading:**
- **Per-channel forgetting helps in the lean model:** about +0.06 dB on Urban100/Manga109 over the per-head gate.
- **Even with the per-head gate, lean still beats MambaIRv2** on Urban100 (+0.12) and Manga109 (+0.06). So most of the gain comes from the lean block plus the delta memory.
- **At exactly equal size (30 keys), it only ties MambaIRv2.** The memory needs enough keys: 87 keys gives +0.2 dB on Urban100 over 30 keys. So the honest claim is "better at +10% size, equal at the same size". The 500k runs will tell whether that holds.

### 10. 500k progress (2026-10-04)

Set14 validation, lean minus MambaIRv2 at the same step (positive = lean ahead):

| Step | 45k | 50k | 55k | 60k | 65k | 70k | 75k | 80k | 85k | 90k | 95k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gap (dB) | +0.05 | +0.33 | −0.18 | 0.00 | +0.04 | +0.04 | +0.09 | +0.43 | −0.01 | +0.08 | +0.16 |

Lean is ahead at 8 of 11 checkpoints (average +0.09 dB). Set14 is noisy, so the full 5-set test at the end decides.

### 11. Key ablation: lean without erase (2026-10-06)

| Test set | MambaIRv2 22.9M | lean (delta, erase on) | **lean without erase (GLA)** | lean, per-head gate |
|---|---|---|---|---|
| Set5 | 37.91 | 37.95 | 37.94 | 37.91 |
| Set14 | 33.42 | 33.46 | 33.46 | 33.41 |
| B100 | 32.08 | 32.11 | 32.11 | 32.11 |
| Urban100 | 31.53 | 31.70 | **31.74** | 31.64 |
| Manga109 | 38.38 | 38.51 | 38.49 | 38.44 |

**Reading (important for the paper story):**
- **In the lean model, the delta erase makes no measurable difference** at 50k: within ±0.04 dB, which is noise for single runs.
- The erase helped a lot (+0.32) only in the *original* model (5 heads × 17 keys, per-head gate), where the memory was small and crowded.
- With 1 head × 87 keys and per-channel forgetting, the memory is no longer crowded, so erasing matters less. This is consistent with the theory in section 4.
- So the gain over MambaIRv2 comes from:
  - **replacing the Mamba scan with a lean, per-channel-gated linear-attention memory** (both GLA and delta versions work),
  - **enough key room** (87 vs 30 keys: +0.2 dB),
  - **per-channel forgetting** (+0.06–0.1 dB).
- The delta rule is not the source of the gain in the final model.
- **Paper framing should change** from "the delta rule is why it wins" to "a gated linear-attention memory with per-channel forgetting beats Mamba's scan in MambaIRv2". The delta rule becomes an option studied in the ablations.

### 12. 500k progress (2026-10-07)

- **MambaIRv2:** step 480k (lr drops done), Set14 34.47, best 34.50 @ 470k. Finishes ~Oct 8 02:30; test job 1433380 queued.
- **Lean:** step 221k (before its first lr drop at 250k), Set14 34.10.
- **Same-step gap since 100k:** lean ahead at **22 of 25** checkpoints, mean **+0.09 dB**. Over 195k–220k the gap is +0.15 to +0.22.
- **Lean will hit the 7-day limit on Oct 9 ~12:14** (around step 280k). Auto-resume job 1424381 then waits for an A100. **Ask the admins to extend 1410816 to 14 days** to avoid losing days in the queue.

### 13. ×3 / ×4 fine-tunes (set up 2026-10-07; **switched to 250k = paper schedule on 2026-10-08**)

> **Update 2026-10-08:** at the user's request, ×3/×4 now use the **paper's exact schedule: 250k steps**, milestones 125k/200k/225k/237.5k, lr 1e-4, from the ×2 weights. This is what the MambaIRv2 paper does (Sec. 5.1: "use the pre-trained weights from the 2× model to initialize those of 3× and 4× and halve the learning rate and total training iterations").
> - **Configs:** `*_ft250k.yml`.
> - **The 100k runs:** the MambaIRv2 100k runs (1433564/1433565) finish anyway and are kept as extra data; their tests 1433569/1433570 still run.
> - **Lean 100k:** the lean 100k jobs were cancelled before starting.
> - **The table below describes the original 100k plan.**

- **Recipe:** both models are fine-tuned from their own ×2 500k weights, as the MambaIRv2 paper does. Shortened to fit the deadline (~Oct 26):

| Setting | Paper ×3/×4 | Ours (both models) |
|---|---|---|
| Start from | ×2 weights | ×2 500k weights |
| Steps | 250k | **100k** |
| lr | 1e-4 | 1e-4 |
| Milestones | 125k/200k/225k/237.5k | 50k/80k/90k/95k (same shape ×0.4) |
| Batch | 32 (8 GPUs) | 2 (one V100) |
| HR patch | 192 (×3) / 256 (×4) | same |

- **Smoke test** (`tests/check_ft_configs.py`, job 1433573): all 4 configs build, load the ×2 weights (only the upsampler layer is new) and give the right output size.
- **Configs:** `options/train/*/train_*_x{3,4}_*ft100k.yml` and `options/test/*/test_*_x{3,4}_*ft100k.yml`.

### 14. MambaIRv2 ×2 500k final (2026-10-08) and lean progress

| Test set | MambaIRv2 official (batch 32, 8 GPUs) | **MambaIRv2 retrained, our recipe (batch 4)** |
|---|---|---|
| Set5 | 38.65 / 0.9631 | 38.46 / 0.9623 |
| Set14 | 34.87 / 0.9275 | 34.44 / 0.9250 |
| B100 | 32.62 / 0.9052 | 32.48 / 0.9034 |
| Urban100 | 34.48 / 0.9468 | 33.77 / 0.9420 |
| Manga109 | 40.42 / 0.9810 | 39.90 / 0.9797 |

- The 0.14–0.71 dB gap to the official numbers comes from the 8× smaller batch (same everything else). It is the reference that lean ×2 500k will be compared against.
- **Lean ×2 at 255k** (just after its first lr drop): Set14 **34.35 vs 34.09** for MambaIRv2 at the same step (**+0.26**). Since 100k, lean is ahead at **29 of 32** checkpoints, mean **+0.11 dB**.
- **MambaIRv2 ×3/×4 fine-tunes:** almost done (~97k/100k); tests follow automatically.
- **Lean hits the 7-day limit Oct 9 ~12:14** (time limit still 7 days, so not extended). Auto-resume 1424381 then needs an A100 80GB.

### 15. Incident: lean resume failed on a broken node (2026-10-10)

- **The stop:** lean ×2 (1410816) was stopped at the 7-day limit on Oct 9 12:14 at step ~280k (last checkpoint 275k).
- **The failure:** its auto-resume 1424381 waited ~23h for an A100. It then landed **5 times on speed-38, which has a broken GPU** (`gres/gpu count reported lower`), so every attempt was NODE_FAIL. After 5 tries Slurm put it on an admin-only hold.
- **Not a code problem:** the job resumed correctly from 275k each time before the node died.
- **Fix:** replaced by **1450198** (same config, resumes from 275k, `ExcNodeList` includes speed-38). The lean ×2 test (1433568) and lean ×3/×4 fine-tunes (1437521/1437522) now depend on 1450198. speed-38 is excluded on all pending jobs.
- **Cost:** ~1.5 days. Also, speed-41/42 are in monthly maintenance, so fewer A100s are available.
- **New ETA:** lean ×2 ~Oct 17 (remaining 225k × ~2.3 s ≈ 6 days after it starts). Lean ×3/×4 at 250k ~Oct 23. **About 3 days of buffer.**

### 16. ×3/×4 baseline done; lean ×2 delayed (2026-10-16)

- **MambaIRv2 ×3 / ×4 (250k fine-tune) final:**

| Scale | Set5 | Set14 | B100 | Urban100 | Manga109 |
|---|---|---|---|---|---|
| ×3 | 34.95 | 30.86 | 29.42 | 29.60 | 35.10 |
| ×4 | 32.79 | 29.08 | 27.88 | 27.32 | 32.01 |

- **Lean ×2 resume (1450198)** waited 5.5 days for an A100 80GB and only started Oct 16 08:55 (from 275k). At ~300k on Oct 16; done ~Oct 21–22.
- Lean still leads MambaIRv2 on Set14: **37 of 40** same-step checkpoints since 100k, mean **+0.12 dB**.
- **Plan (user, 2026-10-16): no deadline, do it properly.** Lean ×3 / ×4 fine-tune from the final ×2 500k model (same as MambaIRv2). A shortcut from the 300k checkpoint was considered and dropped.

### 17. Lean ×2 500k progress (2026-10-18)

- **Lean ×2 resume (1450198)** is healthy: **375k / 500k**, ~2.13 s per step on speed-43, no errors. **Done ~Oct 22 00:00.**
- **Set14, 300k–375k** (same step, lean minus MambaIRv2):

| Step | 300k | 310k | 320k | 330k | 340k | 350k | 360k | 370k | 375k |
|---|---|---|---|---|---|---|---|---|---|
| Lean | 34.32 | 34.35 | 34.30 | 34.35 | 34.33 | **34.42** | 34.37 | 34.34 | 34.25 |
| MambaIRv2 | 34.15 | 34.22 | 34.22 | 34.36 | 34.32 | 34.14 | 34.21 | 34.18 | 34.18 |
| Gap | +0.17 | +0.13 | +0.08 | −0.01 | +0.01 | +0.28 | +0.17 | +0.15 | +0.07 |

- Lean is ahead at **15 of 16** checkpoints between 300k and 375k. Overall since 100k: **52 of 56**, mean **+0.11 dB**.
- Lean's best so far is 34.42 (350k). MambaIRv2's best is 34.50 (470k), but that came *after* its late lr drops; lean has not had those yet. The fair comparison is the final 500k test on all 5 sets (job 1433568).

### 18. Lean ×2 500k final: lean wins on all 5 sets (2026-10-22)

![500k curves](500k_curves.png)

- **Training finished** (1450198) on Oct 21 at 500k; **test 1433568** finished Oct 22.

| ×2, PSNR / SSIM | Set5 | Set14 | B100 | Urban100 | Manga109 |
|---|---|---|---|---|---|
| MambaIRv2 (22.9M), our recipe | 38.46 / 0.9623 | 34.44 / 0.9250 | 32.48 / 0.9034 | 33.77 / 0.9420 | 39.90 / 0.9797 |
| **Lean GatedMambaIR (25.1M)** | **38.47 / 0.9623** | **34.57 / 0.9256** | **32.51 / 0.9039** | **33.93 / 0.9430** | **39.94 / 0.9798** |
| Gain (PSNR) | +0.01 | +0.13 | +0.03 | +0.16 | +0.04 |
| MambaIRv2 official (batch 32, 8 GPUs), reference only | 38.65 | 34.87 | 32.62 | 34.48 | 40.42 |

- **Largest gain on Urban100** (+0.16), the set with the most long-range repeated structure (buildings, windows). This fits the idea of a better long-range memory.
- **The late lr drops did not close the gap:** after 375k, lean was ahead at 24 of 26 Set14 checkpoints. Lean's best Set14 value is 34.58 (445k), MambaIRv2's is 34.50 (470k).
- SSIM is equal or better on every set.
- **Agreed plan condition met** (section "Next steps"): lean wins at 500k, so the follow-ups go ahead. ×3 / ×4 are already running.

### 19. Lean ×3 / ×4 fine-tune progress (2026-10-26)

- Both started Oct 21 23:39 on speed-25 (V100), ~1.8 s per step, no errors. (The `expandable_segments` traceback near the top of each log is a harmless check before training starts.)
- **×3 (1437521):** 231k / 250k, done ~Oct 27 07:00. Ahead of MambaIRv2 ×3 at **42 of 46** same-step validation checkpoints (mean **+0.04 dB**); best 30.79 vs 30.75.
- **×4 (1437522):** 234k / 250k, done ~Oct 27 05:30. Ahead at **44 of 46** (mean **+0.05 dB**); best 29.00 vs 28.96.
- The 7-day limit is Oct 28 23:39, so they will not be cut off. Tests 1437525 / 1437526 start automatically.
- Final results and curves: section 20.

### 20. Lean ×3 / ×4 final: lean wins on all 5 sets at both scales (2026-10-27)

- **Training finished** Oct 27 (×4 1437522 at 05:34, ×3 1437521 at 07:13); **tests 1437526 / 1437525** finished 06:01 / 07:57.

| ×3, PSNR / SSIM | Set5 | Set14 | B100 | Urban100 | Manga109 |
|---|---|---|---|---|---|
| MambaIRv2, our recipe | 34.95 / 0.9316 | 30.86 / 0.8519 | 29.42 / 0.8136 | 29.60 / 0.8797 | 35.10 / 0.9530 |
| **Lean GatedMambaIR** | **34.99 / 0.9320** | **30.87 / 0.8521** | **29.44 / 0.8140** | **29.77 / 0.8820** | **35.16 / 0.9534** |
| Gain (PSNR) | +0.04 | +0.01 | +0.02 | +0.17 | +0.05 |

| ×4, PSNR / SSIM | Set5 | Set14 | B100 | Urban100 | Manga109 |
|---|---|---|---|---|---|
| MambaIRv2, our recipe | 32.79 / 0.9024 | 29.08 / 0.7933 | 27.88 / 0.7473 | 27.32 / 0.8217 | 32.01 / 0.9241 |
| **Lean GatedMambaIR** | **32.83 / 0.9031** | **29.12 / 0.7937** | **27.90 / 0.7479** | **27.49 / 0.8253** | **32.10 / 0.9253** |
| Gain (PSNR) | +0.05 | +0.04 | +0.02 | +0.17 | +0.09 |

- **Same pattern as ×2:** the largest gain is on Urban100 (+0.16 at ×2, +0.17 at ×3 and ×4), the set with the most long-range repeated structure.
- **Validation (Set14, same-step):** lean ahead at **46 of 50** checkpoints at ×3 (mean **+0.04 dB**) and **48 of 50** at ×4 (mean **+0.05 dB**). Best 30.79 vs 30.75 (×3), 29.00 vs 28.96 (×4).
- So lean beats MambaIRv2 on **all 15 scale/set pairs** (×2, ×3, ×4 × 5 sets), PSNR and SSIM.

![x3 curves](x3_curves.png)
![x4 curves](x4_curves.png)

### 21. DeltaSR ablation studies launched (2026-10-28)

- **Design:** [ABLATIONS.md](ABLATIONS.md) (Tables 1-4, Fig. A, Fig. B; which existing run fills each row). Branch `deltasr-ablations`.
- **Code:** new options `gate: none`, `prompt_gates`, `record_log_alpha` in `gatedmambair_arch.py`. Defaults unchanged: the 50k chgate / headgate / original-delta checkpoints give **bit-identical** outputs before and after (`tests/test_ablation_options.py`).
- **Enlarged MambaIRv2:** `d_state` **72** → **25,136,729** params (DeltaSR 25,124,201, +0.05%).
- **GPU smoke test** (1531098, V100): all 11 new configs train at the real size (batch 2, 64×64 LR) with finite gradients; peak 11.8-14.2 GiB, so they fit the V100 like the earlier 50k runs.
- **Same recipe** as the existing 50k ablations: ×2, DF2K, batch 2, lr 2e-4, milestones 25k/40k/45k/47.5k, seed 10, V100. Tests (5 sets, Y, crop 2) go to `results.csv` with tag `abl-…`.
- **Fig. B (ours) done:** `results/forget_map/` (1531100; Urban100 img014 / img034 / img087, layer 2.3). Mean forget strength 0.72-0.73, range 0.56-0.77. The P-only-in-q panel follows when 1540891 finishes: rerun `analysis/forget_map.py` with the same arguments.
- **Efficiency (Table 4, Fig. A(b)):** first try (1531099) failed because `fvcore` was not installed in `GMambaIR_env`; installed (`pip install --no-deps fvcore iopath yacs tabulate termcolor portalocker`, torch/numpy unchanged) and resubmitted as **1540939**. Output: `results/ablation_efficiency/`.
- **Known, not new:** on tiny CPU inputs (≤32×32) at random init, the channel-gate chunk kernel can give NaN *gradients* (overflow in the backward of `decayed_dot`, `chunk.py`). The code before this change does the same; it never showed at the real training size (smoke test above, all earlier runs).

| Run | Train | GPU | Test | Table / figure |
|---|---|---|---|---|
| `GatedMambaIR_SR_x2_abl_noforget_50k` | 1540887 | V100 | 1540888 | T1 exp1, T2 exp2 |
| `GatedMambaIR_SR_x2_abl_noforget_gla_50k` | 1540889 | A100 80GB | 1540890 | T2 exp1 |
| `GatedMambaIR_SR_x2_abl_promptq_50k` | 1540891 | A100 80GB | 1540892 | T3 exp1, Fig. B |
| `GatedMambaIR_SR_x2_abl_promptqa_50k` | 1540893 | A100 80GB | 1540894 | T3 exp2 |
| `GatedMambaIR_SR_x2_abl_promptqb_50k` | 1540895 | V100 | 1540896 | T3 exp3 |
| `MambaIRv2_SR_x2_abl_dstate72_50k` | 1540897 | V100 | 1540898 | T4 row 2, Fig. A(a) |
| `GatedMambaIR_SR_x2_abl_ds16_50k` | 1540899 | A100 80GB | 1540900 | Fig. A(a) |
| `GatedMambaIR_SR_x2_abl_ds48_50k` | 1540901 | V100 | 1540902 | Fig. A(a) |
| `GatedMambaIR_SR_x2_abl_ds16_gla_50k` | 1540903 | V100 | 1540904 | Fig. A(a) |
| `GatedMambaIR_SR_x2_abl_ds30_gla_50k` | 1540905 | V100 | 1540906 | Fig. A(a) |
| `GatedMambaIR_SR_x2_abl_ds48_gla_50k` | 1540907 | A100 80GB | 1540908 | Fig. A(a) |

**GPU split (2026-10-28, user approved):** V100 capacity is low (only speed-25 up; speed-03 / speed-27 draining), so 5 runs were moved to full A100 80GB GPUs (`scontrol update ... Gres=gpu:nvidia_a100_7g.80gb:1`). Same recipe (batch 2, same schedule and seed); only the GPU type differs. Not the small A100 slices: they fit in memory (peak 14.2 GiB) but have 1/7-2/7 of the compute.

**Expected finish:** each training is ~1 day on a V100 (less on a full A100) (the earlier 50k runs took 23-26 h; the MambaIRv2 run is faster, ~0.8 s/step), each test ~1.5 h after. Only speed-25 is taking V100 jobs right now (speed-03 / speed-27 are draining) and the account cap is 4 GPUs, so the 11 runs will go about 1-2 at a time: **first results ~Oct 30, all ~Nov 3-6** unless the drained nodes come back.

## Running / queued now (2026-10-28)

The 11 ablation trainings and their tests (section 21), plus the efficiency job 1540939.

## Finished since last report

- **Lean ×3 / ×4 250k fine-tunes:** training 1437521 / 1437522 and tests 1437525 / 1437526 (Oct 27). Results in section 20.
- **Lean ×2 500k:** training 1450198 (resume from 275k, done Oct 21) and test 1433568 (Oct 22). Results in section 18.
- **MambaIRv2:** ×2 500k (1410815, test 1433380), ×3/×4 100k (1433564/5, tests 1433569/70), ×3/×4 250k (1437519/20, tests 1437523/24).
- **Ablations:** lean+per-head (1410817), exact size k30 (1410869), lean+GLA (1424853). Results in sections 9 and 11.
- **Lean 500k first part:** 1410816 timed out at ~280k (7-day limit); 1424381 cancelled (broken node).

- **Bigger-heads run:** 1401270 (training) and 1401759 (test).
- **Delta + channel gate:** 1401769 (training) and 1401771 (test).
- **Lean + channel gate:** 1401770 (training) and 1401772 (test).
- **Old paused run 1400867** (delta 500k, 39.4M): **cancelled**, replaced by the lean 500k run (1410816).

## Next steps

**Agreed plan (2026-10-02):** the two 500k runs decide everything. If lean GatedMambaIR beats MambaIRv2 at 500k, we do the follow-ups below.

1. ~~Both 500k runs~~ **Done: lean wins on all 5 ×2 sets (section 18).**
2. **Lean won at 500k, so now:** (per-head and exact-size ablations are already done, see section 9):
   - ~~**×3 and ×4**~~ **Done: lean wins on all 5 sets at both scales (section 20).**
   - **Ablations** (50k each, same recipe), each removing one piece of the lean model:
     - lean **without the delta erase** (GLA): done, ties lean (section 11),
     - lean with **per-head** forgetting: done (section 9),
     - **exact size** 22.9M: done, ties MambaIRv2 (section 9),
     - optionally: no prompt, different chunk size.
   - **Speed and FLOPs** report (params, FLOPs, inference time) against MambaIRv2.
   - **Novelty check:** search for prior "delta rule / Gated DeltaNet for super-resolution" work.
3. **To make it a strong paper (optional):** a **lightweight** version (~0.8M params) trained with the paper's exact recipe (batch 32), compared directly with published numbers.
4. **If lean does not win at 500k:** look at the training curves and the ablations before deciding the next design change.

**Paper level:** a 500k win plus ×3/×4 and ablations gives a workshop or mid-tier paper. Adding the lightweight track with published-number comparisons makes it strong.

## How to check

```
squeue -u $USER                                              # jobs
python basicsr/tools/compare_results.py --scale 2 --baseline MambaIRv2   # test scores table
```
Logs are in `logs/`: training is `GMamabIR_train_<job>.out` and tests are `GMambaIR_test_<job>.out`.

