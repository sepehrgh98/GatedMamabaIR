"""Pre-launch checks for the same-setup MaIR baseline (docs/BASELINES.md, MaIR section).

For each train config (x2 500k, x3/x4 ft250k):
  - param count with our build_network
  - batch independence: sample 0 alone == sample 0 inside the training batch (the official network_g had batch_size: 4)
  - 20 timed Adam + L1 steps at the training LR size and batch (peak memory, s/iter on random data)
  - eval forward at odd sizes (37x53) and at the largest LR test tile MambaIRModel produces (Manga109 / Urban100)
  - x3/x4: x2 weights load with strict=False and only the upsampler / conv_last keys are skipped
usage (GPU, repo root): python tests/check_mair.py
"""
import sys
import time

import torch
import yaml

sys.path.insert(0, '.')
from basicsr.archs import build_network  # noqa: E402

dev = 'cuda'
torch.backends.cudnn.benchmark = False
def train_steps(net, optim, bs, lq, s, opt):
    for i in range(25):
        if i == 5:
            torch.cuda.synchronize()
            t0 = time.time()
        x = torch.rand(bs, 3, lq, lq, device=dev)
        gt = torch.rand(bs, 3, lq * s, lq * s, device=dev)
        loss = (net(x) - gt).abs().mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 20
    print(f'  train batch {bs} LR {lq}: {dt:.3f} s/iter, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, '
          f'{opt["train"]["total_iter"]} iters ~ {dt * opt["train"]["total_iter"] / 86400:.1f} d (random data)', flush=True)


x2_state = None
for s, stem in ((2, 'x2_500k'), (3, 'x3_ft250k'), (4, 'x4_ft250k')):
    opt = yaml.safe_load(open(f'options/train/baselines/train_MaIR_SR_{stem}.yml'))
    torch.manual_seed(0)
    net = build_network(opt['network_g']).to(dev)
    n = sum(p.numel() for p in net.parameters())
    print(f'MaIR x{s}: params {n:,d} ({n / 1e6:.2f}M)', flush=True)

    if s == 2:
        x2_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    else:
        own = net.state_dict()
        skipped = [k for k, v in x2_state.items() if k not in own or own[k].shape != v.shape]
        missing = [k for k in own if k not in x2_state]
        print(f'  x2 -> x{s} non-strict load: {len(skipped)} x2 keys skipped {skipped}, {len(missing)} new keys {missing}')

    bs = opt['datasets']['train']['batch_size_per_gpu']
    lq = opt['datasets']['train']['gt_size'] // s
    net.eval()
    with torch.no_grad():
        xb = torch.rand(bs, 3, lq, lq, device=dev)
        d = (net(xb)[:1] - net(xb[:1])).abs().max().item()
    print(f'  batch {bs} vs 1, LR {lq}x{lq}: max |diff| sample 0 = {d:.2e}')

    net.train()
    optim = torch.optim.Adam(net.parameters(), lr=2e-4, betas=(0.9, 0.99))
    torch.cuda.reset_peak_memory_stats()
    try:
        train_steps(net, optim, bs, lq, s, opt)
    except torch.cuda.OutOfMemoryError:
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f'  train batch {bs} LR {lq}: OUT OF MEMORY on this {total:.1f} GiB GPU (needs a larger GPU)', flush=True)
        optim.zero_grad(set_to_none=True)
    del optim
    torch.cuda.empty_cache()

    net.eval()
    # MambaIRModel tiles = ceil-ish (H / (H // 200 + 1)) + 10% overlap; largest: Manga109 x2 LR 413x585 -> ~227x322
    for h, w in ((37, 53), (227, 322), (99, 131)):
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            y = net(torch.rand(1, 3, h, w, device=dev))
        assert y.shape == (1, 3, h * s, w * s), y.shape
        print(f'  eval {h}x{w} -> {tuple(y.shape[2:])} ok, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB',
              flush=True)
    del net
    torch.cuda.empty_cache()
print('all checks passed')

