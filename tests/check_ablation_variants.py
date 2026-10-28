"""GPU smoke test of the DeltaSR ablation training configs (docs/ABLATIONS.md), before launching them.

    sbatch --gres=gpu:1 --constraint=Model-V100 analysis/gpu_job.sh tests/check_ablation_variants.py

For every abl train config: build the network, run a few Adam steps at the real training size (batch 2,
64x64 LR from 128 HR patches) with the config's lr, and check that the loss and all gradients stay finite.
Prints params, peak GPU memory and time per step (V100 32GB must hold it, as the 50k runs used).
"""
import glob
import sys
import time
from os import path as osp

import torch
import torch.nn.functional as F
import yaml

ROOT = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir))
sys.path.insert(0, ROOT)
from basicsr.archs.gatedmambair_arch import GatedMambaIR  # noqa: E402
from basicsr.archs.mambairv2_arch import MambaIRv2  # noqa: E402

ARCHS = {'MambaIRv2': MambaIRv2, 'GatedMambaIR': GatedMambaIR}
STEPS = 4


def main():
    dev = 'cuda'
    print(torch.cuda.get_device_name(0), flush=True)
    cfgs = sorted(glob.glob(osp.join(ROOT, 'options/train/*/train_*_abl_*_50k.yml')))
    assert len(cfgs) == 11, cfgs
    failed = []
    for path in cfgs:
        with open(path) as f:
            opt = yaml.safe_load(f)
        net_opt = dict(opt['network_g'])
        torch.manual_seed(opt['manual_seed'])
        net = ARCHS[net_opt.pop('type')](**net_opt).to(dev).train()
        optim = torch.optim.Adam(net.parameters(), lr=opt['train']['optim_g']['lr'], betas=(0.9, 0.99))
        bs, gt = opt['datasets']['train']['batch_size_per_gpu'], opt['datasets']['train']['gt_size']
        torch.cuda.reset_peak_memory_stats()
        ok, losses, t = True, [], time.time()
        for _ in range(STEPS):
            hr = torch.rand(bs, 3, gt, gt, device=dev)
            lr = F.interpolate(hr, scale_factor=0.5, mode='bicubic', align_corners=False).clamp(0, 1)
            loss = F.l1_loss(net(lr), hr)
            optim.zero_grad()
            loss.backward()
            bad = [n for n, p in net.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
            ok &= torch.isfinite(loss).item() and not bad
            losses.append(loss.item())
            if bad:
                print('   non-finite grads:', bad[:5])
            optim.step()
        torch.cuda.synchronize()
        sec = (time.time() - t) / STEPS
        mem = torch.cuda.max_memory_allocated() / 2 ** 30
        n = sum(p.numel() for p in net.parameters())
        print(f"{'PASS' if ok else 'FAIL'} {opt['name']}: {n:,} params, peak {mem:.1f} GiB, {sec:.2f} s/step, "
              f"loss {' '.join(f'{v:.4f}' for v in losses)}", flush=True)
        if not ok:
            failed.append(opt['name'])
        del net, optim
        torch.cuda.empty_cache()
    print('ALL PASS' if not failed else f'FAILED: {failed}')


if __name__ == '__main__':
    main()
