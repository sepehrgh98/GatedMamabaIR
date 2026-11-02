"""Inference time and peak GPU memory of DeltaSR vs input size, for the new Fig. A(b).

For each LR input size in --sizes (square, x2) it times DeltaSR (d_s 87, trained 50k weights) with the
chunk-wise scan at every chunk length in --chunks, and with the recurrent (token-by-token) scan for sizes up to
--recurrent_max (the recurrence takes ~437 s at 256x256 on a V100, so larger sizes take a long time).

Usage (repo root, GPU node):

    sbatch --gres=gpu:1 --constraint=Model-V100 analysis/gpu_job.sh analysis/scaling_vs_size.py
    python analysis/scaling_vs_size.py --sizes 64 128 256 --recurrent_max 128   # quick check

Outputs in --out_dir (default results/scaling_vs_size): scaling.csv, scaling.png, summary.json.
The CSV is written after every measurement, so a job that runs out of time keeps what it finished.
"""
import argparse
import csv
import json
import os
import sys
import time
from os import path as osp

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir))
sys.path.insert(0, ROOT)
sys.path.insert(0, osp.join(ROOT, 'analysis'))
from basicsr.archs import gatedmambair_arch  # noqa: E402
from analysis.ablation_efficiency import OURS, build, recurrent_scan  # noqa: E402
from analysis.memory_heatmap import route_mode  # noqa: E402

FIELDS = ['size', 'pixels', 'form', 'L', 'time_s', 'peak_mem_mib', 'status']


def measure(net, x, warmup, repeat):
    """Median time (s) and peak memory (MiB) of net(x); returns (None, None) on CUDA OOM."""
    try:
        with torch.no_grad():
            for _ in range(warmup):
                net(x)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(repeat):
                torch.cuda.synchronize()
                t = time.perf_counter()
                net(x)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t)
        return float(np.median(times)), torch.cuda.max_memory_allocated() / 2 ** 20
    except (getattr(torch.cuda, "OutOfMemoryError", RuntimeError), RuntimeError) as e:
        if "out of memory" not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        return None, None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sizes', type=int, nargs='+', default=[64, 128, 256, 384, 512])
    p.add_argument('--chunks', type=int, nargs='+', default=[64, 128, 160])
    p.add_argument('--recurrent_max', type=int, default=256, help='largest size timed with the recurrent scan')
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--repeat', type=int, default=5)
    p.add_argument('--out_dir', default='results/scaling_vs_size')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    out = osp.join(ROOT, args.out_dir)
    os.makedirs(out, exist_ok=True)
    dev = args.device
    gpu = torch.cuda.get_device_name(0)
    print('GPU:', gpu, flush=True)
    rows = []

    def save():
        with open(osp.join(out, 'scaling.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)

    with route_mode('argmax'):
        net, trained, _ = build(OURS, dev)
        print('trained weights:', trained, flush=True)
        scans = [m for m in net.modules() if isinstance(m, gatedmambair_arch.GatedDeltaScan)]
        for size in args.sizes:
            x = torch.rand(1, 3, size, size, device=dev, generator=torch.Generator(dev).manual_seed(0))
            for L in args.chunks:
                for s in scans:
                    s.chunk_size = L
                sec, mem = measure(net, x, args.warmup, args.repeat)
                rows.append({'size': size, 'pixels': size * size, 'form': f'chunk L={L}', 'L': L,
                             'time_s': None if sec is None else round(sec, 4),
                             'peak_mem_mib': None if mem is None else round(mem),
                             'status': 'ok' if sec is not None else 'OOM'})
                print(json.dumps(rows[-1]), flush=True)
                save()
            if size <= args.recurrent_max:
                saved = gatedmambair_arch.chunk_gated_delta_rule
                gatedmambair_arch.chunk_gated_delta_rule = recurrent_scan
                try:
                    sec, mem = measure(net, x, 0, 1)  # one pass: it is slow
                finally:
                    gatedmambair_arch.chunk_gated_delta_rule = saved
                rows.append({'size': size, 'pixels': size * size, 'form': 'recurrent', 'L': None,
                             'time_s': None if sec is None else round(sec, 4),
                             'peak_mem_mib': None if mem is None else round(mem),
                             'status': 'ok' if sec is not None else 'OOM'})
                print(json.dumps(rows[-1]), flush=True)
                save()
            torch.cuda.empty_cache()

    with open(osp.join(out, 'summary.json'), 'w') as f:
        json.dump({'gpu': gpu, 'scale': 2, 'trained_weights': trained, 'rows': rows}, f, indent=1)

    # quick-look plot (the paper figure is made separately)
    fig, (ta, ma) = plt.subplots(1, 2, figsize=(10, 3.8))
    for form in [f'chunk L={L}' for L in args.chunks] + ['recurrent']:
        r = [x for x in rows if x['form'] == form and x['status'] == 'ok']
        if not r:
            continue
        ta.plot([x['size'] for x in r], [x['time_s'] for x in r], 'o-', label=form)
        ma.plot([x['size'] for x in r], [x['peak_mem_mib'] for x in r], 'o-', label=form)
    ta.set(xlabel='LR input side (px)', ylabel='inference time (s)', yscale='log')
    ma.set(xlabel='LR input side (px)', ylabel='peak GPU memory (MiB)')
    for ax in (ta, ma):
        ax.grid(alpha=0.3)
        ax.set_xticks(args.sizes)
    ta.legend(frameon=False)
    fig.suptitle(f'DeltaSR x2 vs input size, {gpu}', fontsize=10)
    fig.tight_layout()
    fig.savefig(osp.join(out, 'scaling.png'), dpi=150)
    print('wrote', out)


if __name__ == '__main__':
    main()
