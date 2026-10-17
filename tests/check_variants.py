"""Build GatedMambaIR variants: param counts, old checkpoints still load, fwd+bwd health and speed on GPU.

    python tests/check_variants.py      (submit with: sbatch --gres=gpu:1 tests/test_cfg.sh ... or tests/check_variants.sh)
"""
import sys
import time
from os import path as osp

import torch
import yaml

ROOT = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir))
sys.path.insert(0, ROOT)
from basicsr.archs.gatedmambair_arch import GatedMambaIR, make_multihead_dims  # noqa: E402


def cfg(name, **overrides):
    with open(osp.join(ROOT, f'options/train/gatedmambair/{name}.yml')) as f:
        c = dict(yaml.safe_load(f)['network_g'])
    c.pop('type')
    c.update(overrides)
    return c


def n_params(net):
    return sum(p.numel() for p in net.parameters())


def main():
    dev = 'cuda'
    print(torch.cuda.get_device_name(0))
    # 1) existing configs are unchanged and their checkpoints load strictly
    for name, expected, ckpt in [
            ('train_GatedMambaIR_SR_x2_delta_50k', 39427613, 'GatedMambaIR_SR_x2_delta_50k/models/net_g_50000.pth'),
            ('train_GatedMambaIR_SR_x2_gla_50k', 39427613, 'GatedMambaIR_SR_x2_gla_50k/models/net_g_50000.pth'),
            ('train_GatedMambaIR_SR_x2_delta_bigkey_50k', 43849997,
             'GatedMambaIR_SR_x2_delta_bigkey_50k/models/net_g_45000.pth')]:
        net = GatedMambaIR(**cfg(name))
        assert n_params(net) == expected, (name, n_params(net), expected)
        state = torch.load(osp.join(ROOT, 'experiments', ckpt), map_location='cpu')
        net.load_state_dict(state.get('params_ema', state.get('params', state)), strict=True)
        print(f'PASS {name}: {expected:,} params, {osp.basename(ckpt)} loads strictly')

    # 2) new variants: size and one training step (batch 1, 64x64 LR patch; batch 2 needs >20GB)
    base = 'train_GatedMambaIR_SR_x2_delta_50k'
    variants = {
        'delta (current)': cfg(base),
        'delta + channel gate': cfg(base, gate='channel', chunk_size=128),
    }
    for kr, hd in [(0.25, 29), (0.25, 87), (0.5, 87), (0.5, 58)]:
        dk, dv, h = make_multihead_dims(348, kr, hd)
        variants[f'lean + channel, {h} heads x {dk // h} keys'] = cfg(
            base, lean=True, gate='channel', chunk_size=128, key_ratio=kr, head_dim=hd)
    torch.manual_seed(0)
    x = torch.rand(1, 3, 64, 64, device=dev)
    for label, c in variants.items():
        net = GatedMambaIR(**c).to(dev)
        for m in net.modules():  # open res_scale so the scan gets gradients
            if hasattr(m, 'res_scale'):
                m.res_scale.data.fill_(1.0)
        times = []
        for _ in range(4):
            torch.cuda.synchronize()
            t0 = time.time()
            y = net(x)
            y.mean().backward()
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        scan = net.layers[0].residual_group.layers[0].assm.gated_scan
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        assert torch.isfinite(y).all() and all(torch.isfinite(g).all() for g in grads), label
        assert scan.q_proj.weight.grad.abs().sum() > 0 and scan.a_proj.weight.grad.abs().sum() > 0, label
        mem = torch.cuda.max_memory_allocated() / 2**30
        print(f'PASS {label}: {n_params(net) / 1e6:.2f}M params, fwd+bwd {min(times[1:]):.2f}s, peak {mem:.1f}GB')
        del net, y, grads
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    print('MambaIRv2-B reference: 22.90M params')


if __name__ == '__main__':
    main()
