# Ported from https://github.com/daitao/SAN @ 7c1e61f0d17b58e3400a14e631d9185dde1881b6,
# files TrainCode/model/san.py, TrainCode/model/common.py, TrainCode/model/MPNCOV/python/MPNCOV.py
# Baseline for the same-setup comparison (docs/BASELINES.md). Layers, init and forward are the official ones; changes:
#   - args object replaced by keyword arguments (defaults = official SAN: 20 groups x 10 blocks, 64 feats, reduction 16);
#   - BasicSR takes [0, 1] images, SAN was trained on [0, 255]: input is scaled by img_range=255 and the output divided
#     back, exactly like BasicSR's RCAN/EDSR ports;
#   - Covpool computes the same centred covariance x I_hat x^T in closed form, (1/M) x x^T - (1/M^2) (x 1)(x 1)^T,
#     with autograd instead of building the M x M matrix I_hat (M = H*W), which does not fit in memory on full test images;
#   - unused non-local modes (gaussian/dot_product/concatenation), Triuvec and the custom load_state_dict are dropped.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from basicsr.utils.registry import ARCH_REGISTRY


# ---------------- common.py ----------------
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class MeanShift(nn.Conv2d):

    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False  # as in the official code (does not freeze the parameters)


class Upsampler(nn.Sequential):

    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feat))
                if act:
                    m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if act:
                m.append(act())
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*m)


# ---------------- MPNCOV.py ----------------
def CovpoolLayer(x):
    # == x.bmm(I_hat).bmm(x^T) with I_hat = (1/M) I - (1/M^2) 11^T (official Covpool), without the M x M matrix
    b, c, h, w = x.shape
    m = h * w
    x = x.reshape(b, c, m)
    s = x.sum(dim=2, keepdim=True)
    return x.bmm(x.transpose(1, 2)) / m - s.bmm(s.transpose(1, 2)) / (m * m)


class Sqrtm(Function):

    @staticmethod
    def forward(ctx, input, iterN):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        I3 = 3.0 * torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
        normA = (1.0 / 3.0) * x.mul(I3).sum(dim=1).sum(dim=1)
        A = x.div(normA.view(batchSize, 1, 1).expand_as(x))
        Y = torch.zeros(batchSize, iterN, dim, dim, requires_grad=False, device=x.device)
        Z = torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, iterN, 1, 1)
        if iterN < 2:
            ZY = 0.5 * (I3 - A)
            Y[:, 0, :, :] = A.bmm(ZY)
        else:
            ZY = 0.5 * (I3 - A)
            Y[:, 0, :, :] = A.bmm(ZY)
            Z[:, 0, :, :] = ZY
            for i in range(1, iterN - 1):
                ZY = 0.5 * (I3 - Z[:, i - 1, :, :].bmm(Y[:, i - 1, :, :]))
                Y[:, i, :, :] = Y[:, i - 1, :, :].bmm(ZY)
                Z[:, i, :, :] = ZY.bmm(Z[:, i - 1, :, :])
            ZY = 0.5 * Y[:, iterN - 2, :, :].bmm(I3 - Z[:, iterN - 2, :, :].bmm(Y[:, iterN - 2, :, :]))
        y = ZY * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
        ctx.save_for_backward(input, A, ZY, normA, Y, Z)
        ctx.iterN = iterN
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, A, ZY, normA, Y, Z = ctx.saved_tensors
        iterN = ctx.iterN
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        der_postCom = grad_output * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
        der_postComAux = (grad_output * ZY).sum(dim=1).sum(dim=1).div(2 * torch.sqrt(normA))
        I3 = 3.0 * torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
        if iterN < 2:
            raise NotImplementedError  # the official code references an undefined variable here
        else:
            dldY = 0.5 * (der_postCom.bmm(I3 - Y[:, iterN - 2, :, :].bmm(Z[:, iterN - 2, :, :])) -
                          Z[:, iterN - 2, :, :].bmm(Y[:, iterN - 2, :, :]).bmm(der_postCom))
            dldZ = -0.5 * Y[:, iterN - 2, :, :].bmm(der_postCom).bmm(Y[:, iterN - 2, :, :])
            for i in range(iterN - 3, -1, -1):
                YZ = I3 - Y[:, i, :, :].bmm(Z[:, i, :, :])
                ZY = Z[:, i, :, :].bmm(Y[:, i, :, :])
                dldY_ = 0.5 * (dldY.bmm(YZ) - Z[:, i, :, :].bmm(dldZ).bmm(Z[:, i, :, :]) - ZY.bmm(dldY))
                dldZ_ = 0.5 * (YZ.bmm(dldZ) - Y[:, i, :, :].bmm(dldY).bmm(Y[:, i, :, :]) - dldZ.bmm(ZY))
                dldY = dldY_
                dldZ = dldZ_
            der_NSiter = 0.5 * (dldY.bmm(I3 - A) - dldZ - A.bmm(dldY))
        grad_input = der_NSiter.div(normA.view(batchSize, 1, 1).expand_as(x))
        grad_aux = der_NSiter.mul(x).sum(dim=1).sum(dim=1)
        for i in range(batchSize):
            grad_input[i, :, :] += (der_postComAux[i] - grad_aux[i] / (normA[i] * normA[i])) \
                * torch.ones(dim, device=x.device).diag()
        return grad_input, None


def SqrtmLayer(var, iterN):
    return Sqrtm.apply(var, iterN)


# ---------------- san.py ----------------
class NONLocalBlock2D(nn.Module):
    """_NonLocalBlockND of the official code, dimension=2, mode='embedded_gaussian' (the only mode SAN uses)."""

    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__()
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool = nn.MaxPool2d
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                         padding=0)
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                        padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                             padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
            self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))

    def forward(self, x):
        batch_size, C, H, W = x.shape
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z


## second-order Channel attention (SOCA)
class SOCA(nn.Module):

    def __init__(self, channel, reduction=8):
        super(SOCA, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, C, h, w = x.shape  # x: NxCxHxW
        h1 = 1000
        w1 = 1000
        if h < h1 and w < w1:
            x_sub = x
        elif h < h1 and w > w1:
            W = (w - w1) // 2
            x_sub = x[:, :, :, W:(W + w1)]
        elif w < w1 and h > h1:
            H = (h - h1) // 2
            x_sub = x[:, :, H:H + h1, :]
        else:
            H = (h - h1) // 2
            W = (w - w1) // 2
            x_sub = x[:, :, H:(H + h1), W:(W + w1)]
        ## MPN-COV
        cov_mat = CovpoolLayer(x_sub)  # Global Covariance pooling layer
        cov_mat_sqrt = SqrtmLayer(cov_mat, 5)  # Matrix square root layer (pre-norm, Newton-Schulz 5 iter, post-com.)
        cov_mat_sum = torch.mean(cov_mat_sqrt, 1)
        cov_mat_sum = cov_mat_sum.view(batch_size, C, 1, 1)
        y_cov = self.conv_du(cov_mat_sum)
        return y_cov * x


## self-attention+ channel attention module
class Nonlocal_CA(nn.Module):

    def __init__(self, in_feat=64, inter_feat=32, reduction=8, sub_sample=False, bn_layer=True):
        super(Nonlocal_CA, self).__init__()
        # second-order channel attention
        self.soca = SOCA(in_feat, reduction=reduction)
        # nonlocal module
        self.non_local = (NONLocalBlock2D(in_channels=in_feat, inter_channels=inter_feat, sub_sample=sub_sample,
                                          bn_layer=bn_layer))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ## divide feature map into 4 part
        batch_size, C, H, W = x.shape
        H1 = int(H / 2)
        W1 = int(W / 2)
        nonlocal_feat = torch.zeros_like(x)

        feat_sub_lu = x[:, :, :H1, :W1]
        feat_sub_ld = x[:, :, H1:, :W1]
        feat_sub_ru = x[:, :, :H1, W1:]
        feat_sub_rd = x[:, :, H1:, W1:]

        nonlocal_lu = self.non_local(feat_sub_lu)
        nonlocal_ld = self.non_local(feat_sub_ld)
        nonlocal_ru = self.non_local(feat_sub_ru)
        nonlocal_rd = self.non_local(feat_sub_rd)
        nonlocal_feat[:, :, :H1, :W1] = nonlocal_lu
        nonlocal_feat[:, :, H1:, :W1] = nonlocal_ld
        nonlocal_feat[:, :, :H1, W1:] = nonlocal_ru
        nonlocal_feat[:, :, H1:, W1:] = nonlocal_rd

        return nonlocal_feat


## Residual  Block (RB)
class RB(nn.Module):

    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(inplace=True),
                 res_scale=1, dilation=2):
        super(RB, self).__init__()
        self.gamma1 = 1.0
        self.conv_first = nn.Sequential(conv(n_feat, n_feat, kernel_size, bias=bias), act,
                                        conv(n_feat, n_feat, kernel_size, bias=bias))
        self.res_scale = res_scale

    def forward(self, x):
        y = self.conv_first(x)
        y = y + x
        return y


## Local-source Residual Attention Group (LSRARG)
class LSRAG(nn.Module):

    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(LSRAG, self).__init__()
        self.rcab = nn.ModuleList([
            RB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(inplace=True), res_scale=1)
            for _ in range(n_resblocks)
        ])
        self.soca = (SOCA(n_feat, reduction=reduction))
        self.conv_last = (conv(n_feat, n_feat, kernel_size))
        self.n_resblocks = n_resblocks
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = x
        for i, l in enumerate(self.rcab):
            x = l(x)
        x = self.soca(x)
        x = self.conv_last(x)
        x = x + residual
        return x


## Second-order Channel Attention Network (SAN)
@ARCH_REGISTRY.register()
class SAN(nn.Module):

    def __init__(self, upscale=2, n_resgroups=20, n_resblocks=10, n_feats=64, reduction=16, res_scale=1, n_colors=3,
                 img_range=255., conv=default_conv):
        super(SAN, self).__init__()
        kernel_size = 3
        scale = upscale
        act = nn.ReLU(inplace=True)
        self.img_range = img_range

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(img_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]

        ## share-source skip connection
        self.gamma = nn.Parameter(torch.zeros(1))
        self.n_resgroups = n_resgroups
        self.RG = nn.ModuleList([
            LSRAG(conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale, n_resblocks=n_resblocks)
            for _ in range(n_resgroups)
        ])
        self.conv_last = conv(n_feats, n_feats, kernel_size)

        # define tail module
        modules_tail = [Upsampler(conv, scale, n_feats, act=False), conv(n_feats, n_colors, kernel_size)]

        self.add_mean = MeanShift(img_range, rgb_mean, rgb_std, 1)
        self.non_local = Nonlocal_CA(in_feat=n_feats, inter_feat=n_feats // 8, reduction=8, sub_sample=False,
                                     bn_layer=False)

        self.head = nn.Sequential(*modules_head)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = x * self.img_range
        x = self.sub_mean(x)
        x = self.head(x)

        ## add nonlocal
        xx = self.non_local(x)

        # share-source skip connection
        residual = xx

        ## share-source residual gruop
        for i, l in enumerate(self.RG):
            xx = l(xx) + self.gamma * residual
        ## add nonlocal
        res = self.non_local(xx)
        res = res + x

        x = self.tail(res)
        x = self.add_mean(x)

        return x / self.img_range
