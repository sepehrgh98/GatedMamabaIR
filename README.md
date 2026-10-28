# GatedMambaIR

Image super-resolution with a **gated linear-attention memory** in place of the Mamba selective scan.

GatedMambaIR keeps the whole [MambaIRv2](https://github.com/csguoh/MambaIR) network (window attention, semantic token routing and sorting, ConvFFNs) and swaps only the state-space block. Its scan keeps a matrix-valued memory with a learned **per-channel forgetting gate**, read and written chunk by chunk over the semantically sorted pixel sequence.

![×2 500k training curves and final test gains](docs/500k_curves.png)

## Results

Classic SR, PSNR (dB) / SSIM on the Y channel. Both models use the **same recipe**: ×2 is trained on DF2K for 500k steps (batch 4, one A100). ×3 and ×4 are 250k-step fine-tunes from each model's own ×2 weights (the MambaIRv2 paper's schedule).

**Lean GatedMambaIR beats MambaIRv2 on all five test sets at ×2, ×3 and ×4, in both PSNR and SSIM.**

### ×2

| Model | Params | Set5 | Set14 | B100 | Urban100 | Manga109 |
|---|---|---|---|---|---|---|
| MambaIRv2 (retrained, same recipe) | 22.9M | 38.46 / 0.9623 | 34.44 / 0.9250 | 32.48 / 0.9034 | 33.77 / 0.9420 | 39.90 / 0.9797 |
| **GatedMambaIR (lean)** | 25.1M | **38.47 / 0.9623** | **34.57 / 0.9256** | **32.51 / 0.9039** | **33.93 / 0.9430** | **39.94 / 0.9798** |
| Gain (PSNR) | | +0.01 | +0.13 | +0.03 | +0.16 | +0.04 |

### ×3

| Model | Set5 | Set14 | B100 | Urban100 | Manga109 |
|---|---|---|---|---|---|
| MambaIRv2 (retrained, same recipe) | 34.95 / 0.9316 | 30.86 / 0.8519 | 29.42 / 0.8136 | 29.60 / 0.8797 | 35.10 / 0.9530 |
| **GatedMambaIR (lean)** | **34.99 / 0.9320** | **30.87 / 0.8521** | **29.44 / 0.8140** | **29.77 / 0.8820** | **35.16 / 0.9534** |
| Gain (PSNR) | +0.04 | +0.01 | +0.02 | +0.17 | +0.05 |

### ×4

| Model | Set5 | Set14 | B100 | Urban100 | Manga109 |
|---|---|---|---|---|---|
| MambaIRv2 (retrained, same recipe) | 32.79 / 0.9024 | 29.08 / 0.7933 | 27.88 / 0.7473 | 27.32 / 0.8217 | 32.01 / 0.9241 |
| **GatedMambaIR (lean)** | **32.83 / 0.9031** | **29.12 / 0.7937** | **27.90 / 0.7479** | **27.49 / 0.8253** | **32.10 / 0.9253** |
| Gain (PSNR) | +0.05 | +0.04 | +0.02 | +0.17 | +0.09 |

- The largest gain is on **Urban100** at every scale (+0.16 / +0.17 / +0.17 dB). It is the set with the most long-range repeated structure.
- Set14 validation during training: lean is ahead at 75 of 81 same-step checkpoints after 100k steps at ×2, 46 of 50 at ×3 and 48 of 50 at ×4.
- The official MambaIRv2 numbers are higher (e.g. ×2 Urban100 34.48) because they use batch 32 on 8 GPUs. Here both models use the same smaller batch, so the comparison is like for like.
- Training curves: [×2](docs/500k_curves.png), [×3](docs/x3_curves.png), [×4](docs/x4_curves.png) (made by `tests/plot_curves.py --scale 2|3|4`).

Every test score is logged in [`docs/results.csv`](docs/results.csv). The running experiment log is [`docs/STATUS.md`](docs/STATUS.md).

## Method

The block changed is MambaIRv2's `ASSM`, which becomes `GSSM` in [`basicsr/archs/gatedmambair_arch.py`](basicsr/archs/gatedmambair_arch.py). The scan kernels are in [`basicsr/archs/chunk.py`](basicsr/archs/chunk.py).

1. A router assigns each pixel one of 128 semantic tokens, and pixels are sorted so similar content becomes neighbouring tokens (as in MambaIRv2).
2. A gated scan runs over the sorted sequence with memory `S` (value × key). For each token, `S` decays by a per-key-channel gate `α`, the value is written with strength `β`, and the output is the query read from `S`. With `use_delta: true` the delta rule first erases what `S` already predicts for the key (Gated DeltaNet style).
3. The sequence is processed in chunks (128 tokens) for speed. The chunked kernel is checked against a token-by-token recurrence in `tests/test_gated_delta_scan.py`.
4. The output is unsorted and added back to the input.

The **lean** variant (`lean: true`) is the one in the results table. It uses the input directly as the value, routes the semantic prompt only into the query and gates, drops the inner norm and output projection, and adds Mamba's `D` skip. That keeps it close to MambaIRv2's size (25.1M vs 22.9M).

**What the ablations show** (50k steps each, details in [`docs/STATUS.md`](docs/STATUS.md)):

- The lean design with the per-channel gate is the source of the gain.
- Removing the delta erase (`use_delta: false`, plain gated linear attention) gives the same scores.
- A per-head gate is slightly worse than the per-channel gate.
- An exact 22.9M-parameter version ties MambaIRv2.

Main config options (`network_g` in the YAML):

| Option | Lean ×2 value | Meaning |
|---|---|---|
| `lean` | `true` | lean block (see above) |
| `gate` | `channel` | forgetting gate per key channel (`head` = one per head) |
| `use_delta` | `true` | delta-rule erase before each write |
| `key_ratio`, `head_dim` | `0.25`, `87` | 1 head × 87 key channels |
| `chunk_size` | `128` | scan chunk length (the channel gate needs ≤ 160) |
| `use_prompt` | `true` | add the semantic-token prompt |

## Repository layout

```
basicsr/archs/gatedmambair_arch.py   GatedMambaIR network (GSSM block)
basicsr/archs/chunk.py               chunked gated / delta scan
basicsr/archs/mambairv2_arch.py      MambaIRv2 baseline
basicsr/tools/                       results CSV tools, bicubic LR generation
options/train|test/gatedmambair/     GatedMambaIR configs (×2 500k, 50k ablations, ×3/×4 fine-tunes)
options/train|test/mambairv2/        MambaIRv2 baseline configs
tests/                               scan correctness test, model checks, plotting scripts
docs/                                status report, method notes, results CSV, figures
train.sh, tests/test_cfg.sh          SLURM launchers for training and testing
realDenoising/                       real-image denoising code inherited from MambaIR
```

## Setup

```bash
conda env create -f environment.yaml      # Python 3.8, PyTorch, mamba_ssm, causal-conv1d
conda activate mambair
pip install -e .
```

Data goes under `datasets/` (git-ignored):

- **Training:** DF2K in `datasets/DF2K/DF2K/HR`, with bicubic LR in `LR_bicubic/X{2,3,4}`.
- **Testing:** Set5, Set14, B100, Urban100 and Manga109 in `datasets/SR/<name>/{HR,LR_bicubic/X{2,3,4}}`.
- **LR images:** `basicsr/tools/generate_bicubic_lr.py` creates them.

The configs use absolute paths under `/speed-scratch/se_gham/GatedMambaIR/`. Change `dataroot_*` and `pretrain_network_g` if your copy lives elsewhere.

## Training and testing

On the SLURM cluster:

```bash
sbatch train.sh options/train/gatedmambair/train_GatedMambaIR_SR_x2_lean_chgate_500k.yml
sbatch tests/test_cfg.sh options/test/gatedmambair/test_GatedMambaIR_SR_x2_lean_chgate_500k.yml
```

Without SLURM:

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 \
    basicsr/train.py -opt <train config> --launcher pytorch --auto_resume
python basicsr/test.py -opt <test config>
```

- `train.sh` resumes from the latest saved state automatically.
- ×3 / ×4 configs (`*_ft250k.yml`) start from the ×2 500k weights. They use `strict_load_g: false` because only the upsampler changes shape.
- `basicsr/test.py` appends every score to `docs/results.csv`.

Compare models:

```bash
python basicsr/tools/compare_results.py --scale 2 --baseline MambaIRv2   # or --scale 3 / 4
```

Run the scan test (CPU, a few seconds):

```bash
python tests/test_gated_delta_scan.py
```

## Acknowledgements

Built on [MambaIR / MambaIRv2](https://github.com/csguoh/MambaIR) and [BasicSR](https://github.com/XPixelGroup/BasicSR). The gated delta scan follows [Gated DeltaNet](https://arxiv.org/abs/2412.06464).
