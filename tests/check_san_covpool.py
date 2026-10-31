"""SAN port: closed-form CovpoolLayer == official MPNCOV Covpool (x I_hat x^T), values and gradients, on GPU.

usage: python tests/check_san_covpool.py <path to daitao/SAN>/TrainCode/model/MPNCOV/python
"""
import sys

import torch

sys.path.insert(0, '.')
sys.path.insert(0, sys.argv[1])
import MPNCOV  # noqa: E402  official
from basicsr.archs.san_arch import CovpoolLayer, SqrtmLayer  # noqa: E402

torch.manual_seed(0)
for shape in [(4, 64, 32, 32), (1, 64, 61, 47)]:
    x = (torch.randn(*shape, device='cuda') * 3 + 1).requires_grad_()
    g = torch.randn(shape[0], shape[1], shape[1], device='cuda')
    ref = MPNCOV.SqrtmLayer(MPNCOV.CovpoolLayer(x), 5)
    (gr, ) = torch.autograd.grad((ref * g).sum(), x)
    new = SqrtmLayer(CovpoolLayer(x), 5)
    (gn, ) = torch.autograd.grad((new * g).sum(), x)
    ev = ((new - ref).abs().max() / ref.abs().max()).item()
    eg = ((gn - gr).abs().max() / gr.abs().max()).item()
    print(f'{shape}: rel max err value {ev:.2e} grad {eg:.2e}')
    assert ev < 1e-4 and eg < 1e-3
print('SAN covpool OK')
