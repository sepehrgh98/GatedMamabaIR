"""Checks for the DeltaSR ablation options (docs/ABLATIONS.md): gate 'none', prompt_gates, record_log_alpha.

Runs on CPU (a few minutes; needs the project env and the 50k checkpoints in experiments/):
    python tests/test_ablation_options.py

1) Existing checkpoints give exactly the same output with the new code as with the arch before the change
   (gatedmambair_arch.py at BASE_COMMIT, read with git show).
2) gate none / prompt_gates variants build with the expected modules, run fwd+bwd and round-trip their state dict.
3) record_log_alpha returns a per-pixel map in image order.
"""
import importlib.util
import subprocess
import sys
import tempfile
from os import path as osp

import torch
import yaml

ROOT = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir))
sys.path.insert(0, ROOT)
from basicsr.archs.gatedmambair_arch import GatedMambaIR  # noqa: E402
from basicsr.utils.registry import ARCH_REGISTRY  # noqa: E402

BASE_COMMIT = 'c27189e'  # last commit before the ablation options
EXP = '/speed-scratch/se_gham/GatedMambaIR/experiments'
CHECKPOINTS = [  # config, checkpoint: the default (channel), per-head and original non-lean paths
    ('train_GatedMambaIR_SR_x2_lean_chgate_50k', 'GatedMambaIR_SR_x2_lean_chgate_50k/models/net_g_50000.pth'),
    ('train_GatedMambaIR_SR_x2_lean_headgate_50k', 'GatedMambaIR_SR_x2_lean_headgate_50k/models/net_g_50000.pth'),
    ('train_GatedMambaIR_SR_x2_delta_50k', 'GatedMambaIR_SR_x2_delta_50k/models/net_g_50000.pth'),
]


def cfg(name='train_GatedMambaIR_SR_x2_lean_chgate_50k', **overrides):
    with open(osp.join(ROOT, f'options/train/gatedmambair/{name}.yml')) as f:
        c = dict(yaml.safe_load(f)['network_g'])
    c.pop('type')
    c.update(overrides)
    return c


def old_arch():
    """gatedmambair_arch.py as of BASE_COMMIT, imported as a separate module."""
    src = subprocess.run(['git', '-C', ROOT, 'show', f'{BASE_COMMIT}:basicsr/archs/gatedmambair_arch.py'],
                         check=True, capture_output=True, text=True).stdout
    path = osp.join(tempfile.mkdtemp(), 'gatedmambair_arch_old.py')
    with open(path, 'w') as f:
        f.write(src)
    new_cls = ARCH_REGISTRY._obj_map.pop('GatedMambaIR')  # the old copy registers the same name
    try:
        spec = importlib.util.spec_from_file_location('gatedmambair_arch_old', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        ARCH_REGISTRY._obj_map['GatedMambaIR'] = new_cls
    return module


def load(net, ckpt):
    state = torch.load(osp.join(EXP, ckpt), map_location='cpu')
    net.load_state_dict(state.get('params_ema', state.get('params', state)), strict=True)
    return net.eval()


def run(net, x):
    torch.manual_seed(0)  # the router's gumbel noise
    with torch.no_grad():
        return net(x)


def scans(net):
    return [m for m in net.modules() if type(m).__name__ == 'GatedDeltaScan']


def test_checkpoints_unchanged():
    old = old_arch()
    x = torch.rand(1, 3, 40, 40, generator=torch.Generator().manual_seed(1))
    for name, ckpt in CHECKPOINTS:
        if not osp.exists(osp.join(EXP, ckpt)):
            print(f'SKIP {name}: no checkpoint')
            continue
        y_new = run(load(GatedMambaIR(**cfg(name)), ckpt), x)
        y_old = run(load(old.GatedMambaIR(**cfg(name)), ckpt), x)
        diff = (y_new - y_old).abs().max().item()
        assert diff == 0, (name, diff)
        print(f'  {name}: identical output (max diff {diff})')


def test_variants_build_and_train():
    expected = [  # overrides, (has a_proj, has ap_proj, has bp_proj)
        ({'gate': 'none'}, (False, False, True)),
        ({'gate': 'none', 'use_delta': False}, (False, False, True)),
        ({'prompt_gates': []}, (True, False, False)),
        ({'prompt_gates': ['a']}, (True, True, False)),
        ({'prompt_gates': ['b']}, (True, False, True)),
        ({'gate': 'head', 'prompt_gates': []}, (True, False, False)),
    ]
    x = torch.rand(1, 3, 24, 28)
    for overrides, (has_a, has_ap, has_bp) in expected:
        net = GatedMambaIR(**cfg(**overrides))
        for s in scans(net):
            assert (hasattr(s, 'a_proj'), hasattr(s, 'ap_proj'), hasattr(s, 'bp_proj')) == (has_a, has_ap, has_bp)
        y = net(x)
        y.mean().backward()
        assert y.shape == (1, 3, 48, 56) and torch.isfinite(y).all(), overrides
        # only check that gradients reach the scan: at random init on tiny CPU inputs the channel-gate chunk kernel
        # can give non-finite grads (same before this change); tests/check_ablation_variants.py checks the real size
        assert all(s.b_proj.weight.grad is not None for s in scans(net)), overrides
        GatedMambaIR(**cfg(**overrides)).load_state_dict(net.state_dict(), strict=True)
        print(f'  {overrides}: {sum(p.numel() for p in net.parameters()):,} params, fwd+bwd ok')


def test_record_log_alpha():
    for overrides, n_gate in (({}, 87), ({'gate': 'head'}, 1), ({'gate': 'none'}, 1)):
        net = GatedMambaIR(**cfg(**overrides)).eval()
        s = scans(net)[7]
        s.record_log_alpha = True
        run(net, torch.rand(1, 3, 20, 24))
        la = s.last_log_alpha
        assert la.shape == (1, 32, 32, n_gate), (overrides, la.shape)  # padded to the 16-pixel window
        assert (la <= 0).all() and (la >= -1 if overrides == {} else la > -float('inf')).all()
        if overrides.get('gate') == 'none':
            assert (la == 0).all()
        assert all(t.last_log_alpha is None for t in scans(net) if t is not s)  # only the chosen layer records
    # training mode never records
    net = GatedMambaIR(**cfg())
    scans(net)[0].record_log_alpha = True
    net(torch.rand(1, 3, 16, 16))
    assert scans(net)[0].last_log_alpha is None


def test_record_log_alpha_is_image_order():
    """The recorded map must equal the gate computed directly on the un-sorted (image-order) tokens."""
    net = GatedMambaIR(**cfg()).eval()
    s = scans(net)[3]
    gssm = [m for m in net.modules() if type(m).__name__ == 'GSSM'][3]
    seen = {}
    s.register_forward_pre_hook(lambda m, inp: seen.update(x=inp[0], p=inp[1]))
    orig_sort = torch.sort

    def spy_sort(*a, **k):  # capture the sort permutation GSSM uses
        out = orig_sort(*a, **k)
        seen['idx'] = out[1]
        return out
    gssm_forward = gssm.forward

    def forward(*a, **k):
        torch.sort = spy_sort
        try:
            return gssm_forward(*a, **k)
        finally:
            torch.sort = orig_sort
    gssm.forward = forward
    s.record_log_alpha = True
    run(net, torch.rand(1, 3, 32, 32))
    direct_sorted = -torch.sigmoid(s.a_proj(seen['x']) + s.ap_proj(seen['p']))  # scan order
    direct = torch.empty_like(direct_sorted)
    direct[:, seen['idx'][0]] = direct_sorted  # sorted position i holds pixel idx[i]
    assert torch.allclose(s.last_log_alpha, direct.view(1, 32, 32, -1), atol=1e-6)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'PASS {name}')
