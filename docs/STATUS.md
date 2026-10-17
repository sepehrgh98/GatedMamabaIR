# GatedMambaIR: Status Report

_Last updated: 2026-10-16 23:50 EDT. Method and background: [GatedMambaIR.md](GatedMambaIR.md). All test scores: [results.csv](results.csv)._

## Summary

- **The size-matched "lean" GatedMambaIR (25.1M params) beats MambaIRv2 (22.9M) on all five ×2 benchmarks** after the same 50k-step training:
  - **+0.17 dB on Urban100**
  - **+0.12 dB on Manga109**
  - +0.03 to +0.04 dB on Set5, Set14 and B100
- That is now the main candidate. Its **full 500k run** is queued next to the MambaIRv2 500k baseline (same recipe).
- The delta erase itself is confirmed: +0.32 dB over GLA on Urban100.
- Per-channel forgetting on the *big* model hurt. On the lean model it is part of the winning combination, so a check run (lean without it) is queued to tell which change matters.

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

## Running / queued now (2026-10-16)

| Job | What | Status |
|---|---|---|
| 1450198 | **Lean ×2 500k** (resume from 275k) | running, ~300k, done ~Oct 21–22 |
| 1433568 | Test of lean ×2 500k, all 5 sets | after 1450198 |
| 1437521 / 1437522 | **Lean ×3 / ×4 fine-tune 250k from ×2 500k** | start after 1450198 (~Oct 22); ~5.4 days each → done ~Oct 28 |
| 1437525 / 1437526 | Tests of the from-500k ×3 / ×4 | after their fine-tunes |

## Finished since last report

- **MambaIRv2:** ×2 500k (1410815, test 1433380), ×3/×4 100k (1433564/5, tests 1433569/70), ×3/×4 250k (1437519/20, tests 1437523/24).
- **Ablations:** lean+per-head (1410817), exact size k30 (1410869), lean+GLA (1424853). Results in sections 9 and 11.
- **Lean 500k first part:** 1410816 timed out at ~280k (7-day limit); 1424381 cancelled (broken node).

- **Bigger-heads run:** 1401270 (training) and 1401759 (test).
- **Delta + channel gate:** 1401769 (training) and 1401771 (test).
- **Lean + channel gate:** 1401770 (training) and 1401772 (test).
- **Old paused run 1400867** (delta 500k, 39.4M): **cancelled**, replaced by the lean 500k run (1410816).

## Next steps

**Agreed plan (2026-10-02):** the two 500k runs decide everything. If lean GatedMambaIR beats MambaIRv2 at 500k, we do the follow-ups below.

1. **Now (queued):** MambaIRv2 500k (1410815) and lean GatedMambaIR 500k (1410816), same recipe. Plus two short 50k checks: per-head gate (1410817) and exact size 22.9M (1410869).
2. **If lean wins at 500k** (per-head and exact-size ablations are already done, see section 9):
   - **×3 and ×4:** fine-tune both models from their ×2 weights (as MambaIRv2 does). Test all 5 sets.
   - **Ablations** (50k each, same recipe), each removing one piece of the lean model:
     - lean **without the delta erase** (GLA): proves the gain comes from the idea,
     - lean with **per-head** forgetting: already queued (1410817),
     - **exact size** 22.9M: already queued (1410869),
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

