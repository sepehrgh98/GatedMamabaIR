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

    opt['val']['test_chop'] (optional): {shave: 10, min_size: 160000}
        split into 4 overlapping quarters, recursively while a quarter has >= min_size LR pixels, as SAN's official
        test code (TestCode/code/model/__init__.py, forward_chop, run with --chop). Needed for SAN: its non-local
        attention on whole Urban100/Manga109 x2 images does not fit in 32GB.
    """

    def _forward_chop(self, net, x, shave, min_size):
        # copy of SAN's forward_chop with n_GPUs = 1
        scale = self.opt.get('scale', 1)
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]
        if w_size * h_size < min_size:
            sr_list = [net(patch) for patch in lr_list]
        else:
            sr_list = [self._forward_chop(net, patch, shave, min_size) for patch in lr_list]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
        return output

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
        chop = self.opt['val'].get('test_chop')
        with torch.no_grad():
            if chop and not self.is_train:
                self.output = self._forward_chop(net, img, chop['shave'], chop['min_size'])
            else:
                self.output = net(img)
        if not hasattr(self, 'net_g_ema'):
            self.net_g.train()
        self.output = self.output[:, :, :h * scale, :w * scale]
