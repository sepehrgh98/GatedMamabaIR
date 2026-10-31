"""Param counts + one forward/backward for every same-setup baseline config (docs/BASELINES.md).

usage: python tests/check_baselines.py [--forward]   (run from the repo root; --forward needs a GPU for MambaIR)
"""
import argparse
import glob
import sys

import torch
import yaml

sys.path.insert(0, '.')
from basicsr.archs import build_network  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--forward', action='store_true')
args = p.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

for cfg in sorted(glob.glob('options/train/baselines/train_*.yml')):
    opt = yaml.safe_load(open(cfg))
    net = build_network(opt['network_g'])
    n = sum(p.numel() for p in net.parameters())
    msg = f'{opt["name"]:24s} params {n:>11,d} ({n / 1e6:.2f}M)'
    if args.forward:
        s = opt['scale']
        lq_size = opt['datasets']['train']['gt_size'] // s
        net = net.to(dev).train()
        x = torch.rand(2, 3, lq_size, lq_size, device=dev)
        y = net(x)
        assert y.shape == (2, 3, lq_size * s, lq_size * s), y.shape
        y.abs().mean().backward()
        unused = [k for k, p in net.named_parameters() if p.grad is None]
        net.eval()
        with torch.no_grad():  # odd test size, padded as BaselineSRModel does
            h, w = 37, 53
            pad = opt['val'].get('test_pad')
            m = pad['multiple'] if pad else 1
            xt = torch.nn.functional.pad(torch.rand(1, 3, h, w, device=dev), (0, (m - w % m) % m, 0, (m - h % m) % m),
                                         'reflect')
            yt = net(xt)[:, :, :h * s, :w * s]
            assert yt.shape == (1, 3, h * s, w * s), yt.shape
        msg += f'  fwd/bwd ok, unused params: {len(unused)}' + (f' {unused[:4]}' if unused else '')
        net.cpu()
        del net
        if dev == 'cuda':
            torch.cuda.empty_cache()
    print(msg, flush=True)
