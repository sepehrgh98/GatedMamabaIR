import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel


@MODEL_REGISTRY.register()
class BaselineSRModel(SRModel):
    """SRModel for the same-setup baselines (docs/BASELINES.md): whole-image test, padded the way each official code does.

    opt['val']['test_pad'] (optional): {multiple: N, mode: reflect | flip}
        reflect: F.pad(..., 'reflect') as HAT's HATModel.pre_process
        flip:    concat with the flipped image as SwinIR's main_test_swinir.py and CAT's CATModel
    No test_pad: the network gets the LR image as is (DAT, RCAN, EDSR, SAN).
    """

    def test(self):
        pad = self.opt['val'].get('test_pad')
        scale = self.opt.get('scale', 1)
        _, _, h, w = self.lq.size()
        mod_pad_h, mod_pad_w = 0, 0
        img = self.lq
        if pad:
            multiple = pad['multiple']
            mod_pad_h = (multiple - h % multiple) % multiple
            mod_pad_w = (multiple - w % multiple) % multiple
            if pad['mode'] == 'reflect':
                img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
            elif pad['mode'] == 'flip':
                img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h + mod_pad_h, :]
                img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w + mod_pad_w]
            else:
                raise ValueError(f'unknown test_pad mode {pad["mode"]}')

        net = self.net_g_ema if hasattr(self, 'net_g_ema') else self.net_g
        net.eval()
        with torch.no_grad():
            self.output = net(img)
        if not hasattr(self, 'net_g_ema'):
            self.net_g.train()
        self.output = self.output[:, :, :h * scale, :w * scale]
