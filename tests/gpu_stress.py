"""GPU checks for GatedMambaIR's scan: gradient health over many seeds + chunk-size speed.

    python tests/gpu_stress.py [--seeds 10] [--size 64]
"""
import argparse
import sys
import time
from os import path as osp

import torch

sys.path.insert(0, osp.dirname(__file__))
from test_gatedmambair_model import build  # noqa: E402


def grad_health(use_delta, seeds, size, batch):
    torch.manual_seed(123)
    net = build(use_delta).cuda()
    for m in net.modules():
        if hasattr(m, 'res_scale'):  # starts at 0; open it so gradients go through the scan
            m.res_scale.data.fill_(1.0)
    bad_runs = 0
    for seed in range(seeds):
        torch.manual_seed(seed)
        net.zero_grad()
        y = net(torch.rand(batch, 3, size, size, device='cuda'))
        (y - torch.rand_like(y)).abs().mean().backward()
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        bad = sum(1 for g in grads if not torch.isfinite(g).all())
        gmax = max(g.abs().max().item() for g in grads)
        bad_runs += bad > 0
        print(f'  use_delta={use_delta} seed {seed}: y finite={torch.isfinite(y).all().item()} '
              f'non-finite grads={bad} max|grad|={gmax:.3g}', flush=True)
    return bad_runs


def speed(use_delta, chunk_size, size, batch, iters=5):
    torch.manual_seed(0)
    net = build(use_delta, chunk_size=chunk_size).cuda()
    x = torch.rand(batch, 3, size, size, device='cuda')
    for _ in range(2):
        net(x).mean().backward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    for _ in range(iters):
        net.zero_grad()
        net(x).mean().backward()
    torch.cuda.synchronize()
    return (time.time() - start) / iters, torch.cuda.max_memory_allocated() / 2**30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--size', type=int, default=64, help='LR patch size (train config: 128 HR / 2)')
    parser.add_argument('--batch', type=int, default=4)
    args = parser.parse_args()
    print(torch.__version__, torch.cuda.get_device_name(0), flush=True)

    print('== gradient health', flush=True)
    summary = {d: grad_health(d, args.seeds, args.size, args.batch) for d in (False, True)}
    for d, n in summary.items():
        print(f'RESULT use_delta={d}: {n}/{args.seeds} runs with non-finite gradients')

    print('== speed (fwd+bwd per training step, batch %d, %dx%d LR)' % (args.batch, args.size, args.size))
    for d in (False, True):
        for cs in (64, 128, 256):
            t, mem = speed(d, cs, args.size, args.batch)
            print(f'RESULT use_delta={d} chunk_size={cs}: {t:.3f} s/iter, peak {mem:.1f} GiB', flush=True)


if __name__ == '__main__':
    main()
