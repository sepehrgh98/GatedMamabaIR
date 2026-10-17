"""Smoke test for the full GatedMambaIR network (needs the project env: mamba_ssm is imported by the arch).

    python tests/test_gatedmambair_model.py [--checkpoint experiments/GatedMambaIR_SR_x2/models/net_g_464000.pth]

Builds the x2 network from its train config with use_delta off and on, checks the parameter count is the
same, runs a forward/backward pass on an odd-sized input, and (if given) loads an old checkpoint strictly.
"""
import argparse
import sys
import time
from os import path as osp

import torch
import yaml

ROOT = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir))
sys.path.insert(0, ROOT)
from basicsr.archs.gatedmambair_arch import GatedMambaIR  # noqa: E402


def build(use_delta, **overrides):
    with open(osp.join(ROOT, 'options/train/gatedmambair/train_GatedMambaIR_SR_x2.yml')) as f:
        cfg = dict(yaml.safe_load(f)['network_g'])
    cfg.pop('type')
    cfg.update(use_delta=use_delta, **overrides)
    return GatedMambaIR(**cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help='old net_g_*.pth to load with strict=True')
    parser.add_argument('--size', type=int, nargs=2, default=[37, 45], help='LR input height width')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)

    counts = {}
    for use_delta in (False, True):
        net = build(use_delta).to(device)
        counts[use_delta] = sum(p.numel() for p in net.parameters())
        # res_scale starts at 0, which would hide the scan; open it so gradients reach the scan
        for m in net.modules():
            if hasattr(m, 'res_scale'):
                m.res_scale.data.fill_(1.0)
        x = torch.rand(1, 3, *args.size, device=device)
        start = time.time()
        y = net(x)
        y.mean().backward()
        elapsed = time.time() - start
        assert y.shape == (1, 3, 2 * args.size[0], 2 * args.size[1]), y.shape
        assert torch.isfinite(y).all()
        scan = net.layers[0].residual_group.layers[0].assm.gated_scan
        assert scan.q_proj.weight.grad is not None and torch.isfinite(scan.q_proj.weight.grad).all()
        print(f'PASS use_delta={use_delta}: out {tuple(y.shape)}, params {counts[use_delta]:,}, '
              f'fwd+bwd {elapsed:.1f}s on {device}')

        if args.checkpoint:
            state = torch.load(args.checkpoint, map_location='cpu')
            state = state.get('params_ema', state.get('params', state))
            build(use_delta).load_state_dict(state, strict=True)
            print(f'PASS use_delta={use_delta}: {osp.basename(args.checkpoint)} loads with strict=True')

    assert counts[False] == counts[True], counts


if __name__ == '__main__':
    main()
