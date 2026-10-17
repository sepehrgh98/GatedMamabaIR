"""Smoke-test the x3/x4 fine-tune configs: build the net, load the x2 weights non-strictly, check output size.

    python tests/check_ft_configs.py   (on a GPU node)
"""
import glob
import sys
from os import path as osp

import torch
import yaml

ROOT = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir))
sys.path.insert(0, ROOT)
from basicsr.archs.gatedmambair_arch import GatedMambaIR  # noqa: E402
from basicsr.archs.mambairv2_arch import MambaIRv2  # noqa: E402

CONFIGS = ['options/train/mambairv2/train_MambaIRv2_SR_x{s}_ft100k.yml',
           'options/train/gatedmambair/train_GatedMambaIR_SR_x{s}_lean_chgate_ft100k.yml']
for tmpl in CONFIGS:
    for s in (3, 4):
        opt = yaml.safe_load(open(osp.join(ROOT, tmpl.format(s=s))))
        cfg = dict(opt['network_g'])
        net = {'GatedMambaIR': GatedMambaIR, 'MambaIRv2': MambaIRv2}[cfg.pop('type')](**cfg).cuda()
        # the final x2 weights do not exist yet: use the newest x2 checkpoint in the same folder
        x2_dir = osp.dirname(opt['path']['pretrain_network_g'])
        ckpt = max(glob.glob(osp.join(x2_dir, 'net_g_[0-9]*.pth')), key=lambda p: int(p.split('_')[-1][:-4]))
        state = torch.load(ckpt, map_location='cpu')
        state = state.get('params_ema', state.get('params', state))
        own = net.state_dict()
        skipped = [k for k in state if k in own and own[k].shape != state[k].shape]
        loadable = {k: v for k, v in state.items() if k not in skipped}
        missing, unexpected = net.load_state_dict(loadable, strict=False)
        x = torch.rand(1, 3, 64, 64, device='cuda')
        y = net(x)
        y.mean().backward()
        assert y.shape == (1, 3, 64 * s, 64 * s), y.shape
        assert torch.isfinite(y).all()
        print(f'PASS {osp.basename(tmpl.format(s=s))}: from {osp.basename(ckpt)}, loaded {len(loadable)}, '
              f'skipped (shape) {skipped}, missing {list(missing)}, unexpected {list(unexpected)}')
        del net, y
        torch.cuda.empty_cache()
