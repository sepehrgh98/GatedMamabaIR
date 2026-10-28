"""Check the chunked gated (delta) scan against a token-by-token recurrence.

Runs on CPU in a few seconds:
    python tests/test_gated_delta_scan.py        (or: pytest tests/test_gated_delta_scan.py)
"""
import importlib.util
from os import path as osp

import torch
import torch.nn.functional as F

# load chunk.py by path: importing basicsr.archs pulls in mamba_ssm and every arch
_spec = importlib.util.spec_from_file_location(
    'chunk', osp.join(osp.dirname(__file__), osp.pardir, 'basicsr', 'archs', 'chunk.py'))
chunk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chunk)

# the model's real head sizes: 5 heads, key 17, value 69
B, H, DK, DV = 2, 5, 17, 69


def make_inputs(L, decay='mixed', seed=0, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    q = F.normalize(torch.randn(B, H, L, DK, generator=g), dim=-1)
    k = F.normalize(torch.randn(B, H, L, DK, generator=g), dim=-1)
    v = torch.randn(B, H, L, DV, generator=g)
    a = torch.randn(B, H, L, generator=g) * 2
    if decay == 'strong':  # alpha around 0.3-0.5: the case where the old cumprod underflowed
        a = a * 0.2 - 0.5
    log_alpha = F.logsigmoid(a)
    beta = torch.sigmoid(torch.randn(B, H, L, generator=g))
    tensors = [q, k, v, log_alpha, beta]
    if requires_grad:
        tensors = [t.clone().requires_grad_(True) for t in tensors]
    return tensors


def max_rel_err(x, ref):
    return ((x - ref).abs().max() / ref.abs().max().clamp_min(1e-12)).item()


def test_forward_matches_recurrent():
    for use_delta in (False, True):
        for L, T in ((100, 16), (130, 64), (64, 64), (300, 128)):
            for decay in ('mixed', 'strong'):
                q, k, v, la, b = make_inputs(L, decay)
                ref_o, ref_S = chunk.gated_delta_rule_recurrent(q, k, v, la, b, use_delta=use_delta)
                o, S = chunk.chunk_gated_delta_rule(q, k, v, la, b, chunk_size=T, use_delta=use_delta)
                err_o, err_S = max_rel_err(o, ref_o), max_rel_err(S, ref_S)
                assert err_o < 1e-4 and err_S < 1e-4, (use_delta, L, T, decay, err_o, err_S)


def test_initial_state_is_carried():
    """Two halves with the state passed between them == one pass over the whole sequence."""
    for use_delta in (False, True):
        q, k, v, la, b = make_inputs(96)
        full, S_full = chunk.chunk_gated_delta_rule(q, k, v, la, b, chunk_size=16, use_delta=use_delta)
        first, S_mid = chunk.chunk_gated_delta_rule(q[:, :, :40], k[:, :, :40], v[:, :, :40], la[:, :, :40],
                                                    b[:, :, :40], chunk_size=16, use_delta=use_delta)
        second, S_end = chunk.chunk_gated_delta_rule(q[:, :, 40:], k[:, :, 40:], v[:, :, 40:], la[:, :, 40:],
                                                     b[:, :, 40:], chunk_size=16, use_delta=use_delta,
                                                     initial_state=S_mid)
        assert max_rel_err(torch.cat([first, second], dim=2), full) < 1e-5
        assert max_rel_err(S_end, S_full) < 1e-5


def test_gradients_match_recurrent():
    for use_delta in (False, True):
        inputs = make_inputs(40, requires_grad=True)
        ref_inputs = [t.detach().clone().requires_grad_(True) for t in inputs]
        weight = torch.randn(B, H, 40, DV, generator=torch.Generator().manual_seed(1))

        o, _ = chunk.chunk_gated_delta_rule(*inputs, chunk_size=16, use_delta=use_delta)
        ref_o, _ = chunk.gated_delta_rule_recurrent(*ref_inputs, use_delta=use_delta)
        (o * weight).sum().backward()
        (ref_o * weight).sum().backward()
        for name, t, r in zip(('q', 'k', 'v', 'log_alpha', 'beta'), inputs, ref_inputs):
            assert torch.isfinite(t.grad).all(), (use_delta, name)
            assert max_rel_err(t.grad, r.grad) < 1e-4, (use_delta, name, max_rel_err(t.grad, r.grad))


def test_old_scan_is_wrong():
    """Documents the bug in the previous chunk_gated_delta_fullrank (no read within the chunk)."""
    T = 64
    q, k, v, la, b = make_inputs(3 * T)
    ref_o, _ = chunk.gated_delta_rule_recurrent(q, k, v, la, b, use_delta=False)
    S = torch.zeros(B, H, DV, DK)
    outs = []
    for s in range(0, 3 * T, T):
        o, S = chunk.chunk_gated_delta_fullrank(q[:, :, s:s + T], k[:, :, s:s + T], v[:, :, s:s + T],
                                                la[:, :, s:s + T].exp(), b[:, :, s:s + T], S)
        outs.append(o)
    old_o = torch.cat(outs, dim=2)
    assert old_o[:, :, :T].abs().max() == 0  # the whole first chunk outputs nothing
    assert max_rel_err(old_o, ref_o) > 0.5


def make_channel_inputs(L, decay='mixed', seed=0, requires_grad=False):
    """Per-key-channel decay, parametrised as in GatedDeltaScan(gate='channel'): log_alpha = -sigmoid(a)."""
    q, k, v, _, beta = make_inputs(L, seed=seed)
    a = torch.randn(B, H, L, DK, generator=torch.Generator().manual_seed(seed + 100)) * 2
    if decay == 'strong':  # every channel near the bound (alpha ~0.37-0.5): the largest chunk exponents
        a = a * 0.2 + 3
    log_alpha = -torch.sigmoid(a)
    tensors = [q, k, v, log_alpha, beta]
    if requires_grad:
        tensors = [t.clone().requires_grad_(True) for t in tensors]
    return tensors


def test_per_channel_forward_matches_recurrent():
    for use_delta in (False, True):
        for L, T in ((100, 16), (130, 64), (300, 128)):
            for decay in ('mixed', 'strong'):
                q, k, v, la, b = make_channel_inputs(L, decay)
                ref_o, ref_S = chunk.gated_delta_rule_recurrent(q, k, v, la, b, use_delta=use_delta)
                o, S = chunk.chunk_gated_delta_rule(q, k, v, la, b, chunk_size=T, use_delta=use_delta)
                assert torch.isfinite(o).all() and torch.isfinite(S).all(), (use_delta, L, T, decay)
                err_o, err_S = max_rel_err(o, ref_o), max_rel_err(S, ref_S)
                assert err_o < 1e-4 and err_S < 1e-4, (use_delta, L, T, decay, err_o, err_S)


def test_per_channel_gradients_match_recurrent():
    for use_delta in (False, True):
        for decay, T in (('mixed', 16), ('strong', 128)):
            L = 40 if T == 16 else 150
            inputs = make_channel_inputs(L, decay, requires_grad=True)
            ref_inputs = [t.detach().clone().requires_grad_(True) for t in inputs]
            weight = torch.randn(B, H, L, DV, generator=torch.Generator().manual_seed(1))

            o, _ = chunk.chunk_gated_delta_rule(*inputs, chunk_size=T, use_delta=use_delta)
            ref_o, _ = chunk.gated_delta_rule_recurrent(*ref_inputs, use_delta=use_delta)
            (o * weight).sum().backward()
            (ref_o * weight).sum().backward()
            for name, t, r in zip(('q', 'k', 'v', 'log_alpha', 'beta'), inputs, ref_inputs):
                assert torch.isfinite(t.grad).all(), (use_delta, decay, name)
                err = max_rel_err(t.grad, r.grad)
                assert err < 1e-4, (use_delta, decay, name, err)


def test_per_channel_with_equal_channels_is_per_head():
    for use_delta in (False, True):
        q, k, v, la, b = make_inputs(130)
        o_head, S_head = chunk.chunk_gated_delta_rule(q, k, v, la, b, chunk_size=64, use_delta=use_delta)
        la_ch = la.unsqueeze(-1).expand(-1, -1, -1, DK).contiguous()
        o_ch, S_ch = chunk.chunk_gated_delta_rule(q, k, v, la_ch, b, chunk_size=64, use_delta=use_delta)
        assert max_rel_err(o_ch, o_head) < 1e-5 and max_rel_err(S_ch, S_head) < 1e-5


def test_zero_decay_matches_recurrent():
    """gate: none passes log_alpha = 0 (alpha = 1, never forget); the chunked form must stay exact and finite."""
    for use_delta in (False, True):
        for L, T in ((300, 128), (130, 64)):
            q, k, v, la, b = make_inputs(L)
            la = torch.zeros_like(la)
            ref_o, ref_S = chunk.gated_delta_rule_recurrent(q, k, v, la, b, use_delta=use_delta)
            o, S = chunk.chunk_gated_delta_rule(q, k, v, la, b, chunk_size=T, use_delta=use_delta)
            assert torch.isfinite(o).all() and torch.isfinite(S).all(), (use_delta, L, T)
            err_o, err_S = max_rel_err(o, ref_o), max_rel_err(S, ref_S)
            assert err_o < 1e-4 and err_S < 1e-4, (use_delta, L, T, err_o, err_S)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f'PASS {name}')
