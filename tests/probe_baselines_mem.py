"""Peak GPU memory and s/iter of one training step (Adam + L1) for each same-setup baseline, at the config's batch and crop.

usage: python tests/probe_baselines_mem.py [Model ...]   (run from the repo root, on the GPU type to check)
"""
import glob
import sys
import time

import torch
import yaml

sys.path.insert(0, '.')
from basicsr.archs import build_network  # noqa: E402

models = sys.argv[1:]
print(torch.cuda.get_device_name(0), flush=True)
for cfg in sorted(glob.glob('options/train/baselines/train_*_SR_x[24]_*.yml')):
    opt = yaml.safe_load(open(cfg))
    if models and opt['name'].split('_SR_')[0] not in models:
        continue
    s = opt['scale']
    tr = opt['datasets']['train']
    b, lq = tr['batch_size_per_gpu'], tr['gt_size'] // s
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        net = build_network(opt['network_g']).cuda().train()
        optim = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=2e-4, betas=(0.9, 0.99))
        x = torch.rand(b, 3, lq, lq, device='cuda')
        gt = torch.rand(b, 3, lq * s, lq * s, device='cuda')
        times = []
        for i in range(25):
            torch.cuda.synchronize()
            t = time.time()
            optim.zero_grad()
            loss = (net(x) - gt).abs().mean()
            loss.backward()
            optim.step()
            torch.cuda.synchronize()
            times.append(time.time() - t)
        it = sum(times[5:]) / len(times[5:])
        mem = torch.cuda.max_memory_allocated() / 2**30
        print(f'{opt["name"]:24s} batch {b} lq {lq}: peak {mem:5.1f} GiB, {it:.3f} s/iter', flush=True)
    except RuntimeError as e:
        print(f'{opt["name"]:24s} batch {b} lq {lq}: FAILED {str(e).splitlines()[0][:100]}', flush=True)
    net = optim = x = gt = loss = None
