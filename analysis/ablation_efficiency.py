"""Efficiency numbers for the DeltaSR ablations (docs/ABLATIONS.md): Table 4 columns and Fig. A(b).

Usage (from the repo root, on a GPU node):

    python analysis/ablation_efficiency.py                       # everything
    python analysis/ablation_efficiency.py --skip_recurrent      # without the slow token-by-token reference

Table 4: params, FLOPs (fvcore, as analysis/flops_param_fvcore.py), inference time and peak GPU memory on one
--size x --size LR input (x2), for MambaIRv2 d_state 16 / 72 and DeltaSR d_s 30 / 87. Timing and memory do not
depend on the weights, so a model whose checkpoint does not exist yet is timed with random weights. Urban100 /
Manga109 PSNR / SSIM are filled in from docs/results.csv when the test has run.

Fig. A(b): DeltaSR (d_s 87, trained 50k weights) with the scan chunk length L in --chunks, plus the recurrent
(token-by-token) form as reference: time and peak memory, and the max output difference to L = 128 on one real
LR image (the chunk-wise form is exact, so it should be at float32 round-off).

Outputs in --out_dir (default results/ablation_efficiency): table4.csv, chunks.csv, chunks.png, summary.json.
"""
import argparse
import csv
import json
import os
import sys
import time
from os import path as osp

import cv2
import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir))
sys.path.insert(0, ROOT)
sys.path.insert(0, osp.join(ROOT, 'analysis'))
from basicsr.archs import chunk as chunk_mod  # noqa: E402
from basicsr.archs import gatedmambair_arch  # noqa: E402
from basicsr.archs.gatedmambair_arch import GatedMambaIR  # noqa: E402
from basicsr.archs.mambairv2_arch import MambaIRv2  # noqa: E402
from analysis.memory_heatmap import route_mode  # noqa: E402
from analysis.utils_fvcore import FLOPs  # noqa: E402

ARCHS = {'MambaIRv2': MambaIRv2, 'GatedMambaIR': GatedMambaIR}
TABLE4 = [  # label, test config
    ('MambaIRv2, d_state 16', 'options/test/mambairv2/test_MambaIRv2_SR_x2_50k.yml'),
    ('MambaIRv2, d_state 72', 'options/test/mambairv2/test_MambaIRv2_SR_x2_abl_dstate72_50k.yml'),
    ('DeltaSR, d_s 30', 'options/test/gatedmambair/test_GatedMambaIR_SR_x2_lean_k30_50k.yml'),
    ('DeltaSR, d_s 87', 'options/test/gatedmambair/test_GatedMambaIR_SR_x2_lean_chgate_50k.yml'),
]
OURS = 'options/test/gatedmambair/test_GatedMambaIR_SR_x2_lean_chgate_50k.yml'


def build(opt_path, device):
    with open(osp.join(ROOT, opt_path)) as f:
        opt = yaml.safe_load(f)
    net_opt = dict(opt['network_g'])
    net = ARCHS[net_opt.pop('type')](**net_opt)
    ckpt = opt['path']['pretrain_network_g']
    trained = osp.exists(ckpt)
    if trained:
        state = torch.load(ckpt, map_location='cpu')
        net.load_state_dict(state.get('params_ema', state.get('params', state)), strict=True)
    return net.to(device).eval(), trained, opt['name']


def flops_of(net, size, device):
    """fvcore FLOPs (as analysis/flops_param_fvcore.py) plus the ops fvcore could not count."""
    from fvcore.nn.flop_count import flop_count
    x = torch.randn(1, 3, size, size, device=device)
    with torch.no_grad():
        gflops, unsupported = flop_count(model=net, inputs=(x,), supported_ops=FLOPs.register_supported_ops())
    return sum(gflops.values()), dict(unsupported)


def time_and_memory(net, x, warmup, repeat):
    with torch.no_grad():
        for _ in range(warmup):
            net(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(repeat):
            torch.cuda.synchronize()
            t = time.perf_counter()
            y = net(x)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
    return float(np.median(times)), torch.cuda.max_memory_allocated() / 2 ** 20, y


def test_scores(name):
    with open(osp.join(ROOT, 'docs/results.csv')) as f:
        rows = [r for r in csv.DictReader(f) if r['name'] == name]
    return {r['dataset']: (float(r['psnr']), float(r['ssim'])) for r in rows}


def recurrent_scan(q, k, v, log_alpha, beta, chunk_size=64, use_delta=True, initial_state=None):
    o, S = chunk_mod.gated_delta_rule_recurrent(q, k, v, log_alpha, beta, use_delta=use_delta,
                                                initial_state=initial_state)
    return o.to(v.dtype), S


def load_lr(path, device):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    assert img is not None, f'cannot read {path}'
    img = img[:, :, ::-1].astype(np.float32) / 255.
    return torch.from_numpy(img.copy()).permute(2, 0, 1)[None].to(device)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--size', type=int, default=256, help='LR input side for time / memory / FLOPs')
    p.add_argument('--chunks', type=int, nargs='+', default=[32, 64, 128, 160])
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--repeat', type=int, default=5)
    p.add_argument('--skip_recurrent', action='store_true', help='skip the slow token-by-token reference')
    p.add_argument('--image', default='datasets/SR/Urban100/LR_bicubic/X2/img004x2.png',
                   help='real LR image for the output check across L (cropped to --size)')
    p.add_argument('--out_dir', default='results/ablation_efficiency')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    os.makedirs(osp.join(ROOT, args.out_dir), exist_ok=True)
    dev = args.device
    gpu = torch.cuda.get_device_name(0)
    print('GPU:', gpu, flush=True)
    x = torch.rand(1, 3, args.size, args.size, device=dev, generator=torch.Generator(dev).manual_seed(0))
    summary = {'gpu': gpu, 'size': args.size, 'scale': 2, 'table4': [], 'chunks': []}

    # ---- Table 4
    with route_mode('argmax'):
        for label, opt_path in TABLE4:
            net, trained, name = build(opt_path, dev)
            params = sum(p.numel() for p in net.parameters())
            gflops, unsupported = flops_of(net, args.size, dev)
            sec, mem, _ = time_and_memory(net, x, args.warmup, args.repeat)
            scores = test_scores(name)
            row = {'model': label, 'params': params, 'gflops': round(gflops, 1), 'time_s': round(sec, 4),
                   'peak_mem_mib': round(mem), 'trained_weights': trained,
                   'urban100': scores.get('Urban100'), 'manga109': scores.get('Manga109'),
                   'fvcore_uncounted_ops': unsupported}
            summary['table4'].append(row)
            print(json.dumps(row), flush=True)
            del net
            torch.cuda.empty_cache()

        # ---- Fig. A(b): chunk length, plus the recurrent form
        net, trained, _ = build(OURS, dev)
        assert trained, 'Fig. A(b) needs the trained DeltaSR 50k weights'
        scans = [m for m in net.modules() if isinstance(m, gatedmambair_arch.GatedDeltaScan)]
        img = load_lr(osp.join(ROOT, args.image), dev)[..., :args.size, :args.size]
        outs = {}
        for L in args.chunks:
            for s in scans:
                s.chunk_size = L
            sec, mem, _ = time_and_memory(net, x, args.warmup, args.repeat)
            with torch.no_grad():
                outs[L] = net(img)
            summary['chunks'].append({'form': f'chunk L={L}', 'L': L, 'time_s': round(sec, 4), 'peak_mem_mib': round(mem)})
            print(json.dumps(summary['chunks'][-1]), flush=True)
        if not args.skip_recurrent:
            saved = gatedmambair_arch.chunk_gated_delta_rule
            gatedmambair_arch.chunk_gated_delta_rule = recurrent_scan
            try:
                sec, mem, _ = time_and_memory(net, x, 0, 1)  # one pass: it is slow
                with torch.no_grad():
                    outs['recurrent'] = net(img)
            finally:
                gatedmambair_arch.chunk_gated_delta_rule = saved
            summary['chunks'].append({'form': 'recurrent', 'L': None, 'time_s': round(sec, 4), 'peak_mem_mib': round(mem)})
            print(json.dumps(summary['chunks'][-1]), flush=True)
        ref = outs[128] if 128 in outs else outs[args.chunks[0]]
        for row, out in zip(summary['chunks'], outs.values()):
            row['max_abs_diff_vs_L128'] = float((out - ref).abs().max())  # outputs in [0, 1] image range
        summary['chunk_check_image'] = args.image

    # ---- write
    out = osp.join(ROOT, args.out_dir)
    with open(osp.join(out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    with open(osp.join(out, 'table4.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'params', 'GFLOPs', 'time_s', 'peak_mem_MiB', 'Urban100 PSNR/SSIM', 'Manga109 PSNR/SSIM',
                    'trained_weights'])
        for r in summary['table4']:
            fmt = (lambda s: f'{s[0]:.2f} / {s[1]:.4f}' if s else 'pending')
            w.writerow([r['model'], r['params'], r['gflops'], r['time_s'], r['peak_mem_mib'], fmt(r['urban100']),
                        fmt(r['manga109']), r['trained_weights']])
    with open(osp.join(out, 'chunks.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['form', 'L', 'time_s', 'peak_mem_mib', 'max_abs_diff_vs_L128'])
        w.writeheader()
        w.writerows(summary['chunks'])

    rows = [r for r in summary['chunks'] if r['L']]
    fig, (ta, ma) = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, ylabel in ((ta, 'time_s', 'inference time (s)'), (ma, 'peak_mem_mib', 'peak GPU memory (MiB)')):
        ax.plot([r['L'] for r in rows], [r[key] for r in rows], 'o-', color='#eb6834', label='chunk-wise')
        rec = [r for r in summary['chunks'] if r['form'] == 'recurrent']
        if rec:
            ax.axhline(rec[0][key], color='#2a78d6', ls='--', label='recurrent (token by token)')
        ax.set(xlabel='chunk length L', ylabel=ylabel, xticks=[r['L'] for r in rows])
        ax.grid(alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
    if any(r['form'] == 'recurrent' for r in summary['chunks']):
        ta.set_yscale('log')
    ta.legend(frameon=False)
    fig.suptitle(f'DeltaSR x2, one {args.size}x{args.size} LR input, {gpu}', fontsize=10)
    fig.tight_layout()
    fig.savefig(osp.join(out, 'chunks.png'), dpi=150)
    print('wrote', out)


if __name__ == '__main__':
    main()
