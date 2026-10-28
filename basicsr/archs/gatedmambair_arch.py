import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange, repeat

from basicsr.archs.chunk import chunk_gated_delta_lowrank, chunk_gated_delta_rule

def l2norm(x, dim=-1, eps=1e-12):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def safe_gumbel_softmax(logits, tau=1.0, hard=False, dim=-1):
    """F.gumbel_softmax, but with uniforms clamped away from 0 and 1.

    F.gumbel_softmax uses exponential_(), which can return an exact 0 in torch 2.0; -log(0) = inf then
    turns the softmax into NaN.
    """
    u = torch.rand_like(logits).clamp_(1e-10, 1 - 1e-7)
    y_soft = ((logits - torch.log(-torch.log(u))) / tau).softmax(dim)
    if not hard:
        return y_soft
    index = y_soft.argmax(dim, keepdim=True)
    y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
    return y_hard - y_soft.detach() + y_soft


def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r


def semantic_neighbor(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x


class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x


class Gate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)  # DW Conv

    def forward(self, x, H, W):
        x1, x2 = x.chunk(2, dim=-1)
        B, N, C = x.shape
        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C // 2, H, W)).flatten(2).transpose(-1, -2).contiguous()
        return x1 * x2


class GatedMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = Gate(hidden_features // 2)
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, x_size):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        H, W = x_size
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x, H, W)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    r"""
    Shifted Window-based Multi-head Self-Attention

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.proj = nn.Linear(dim, dim)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, rpi, mask=None):
        r"""
        Args:
            qkv: Input query, key, and value tokens with shape of (num_windows*b, n, c*3)
            rpi: Relative position index
            mask (0/-inf):  Mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c3 = qkv.shape
        c = c3 // 3
        qkv = qkv.reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}, qkv_bias={self.qkv_bias}'


def make_multihead_dims(hidden, key_ratio=0.25, head_dim=16):
    """
    Derive (dk, dv, num_heads) from hidden dimension
    while guaranteeing divisibility.
    """
    dk = int(hidden * key_ratio)
    dk = max(head_dim, dk)

    num_heads = max(1, dk // head_dim)

    # snap dimensions to be divisible
    dk = (dk // num_heads) * num_heads
    dv = (hidden // num_heads) * num_heads

    return dk, dv, num_heads



class GSSM(nn.Module):
    def __init__(self, 
        dim, 
        d_state, 
        input_resolution, 
        num_tokens=64, 
        inner_rank=128, 
        mlp_ratio=2.,
        key_ratio=0.25,
        head_dim=16,
        use_prompt=True,
        use_lowrank=False,
        rank=16,
        use_delta=False,
        chunk_size=64,
        gate='head',
        lean=False,
        prompt_gates=('a', 'b'),
    ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank

        # Mamba params
        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)
        self.d_state = d_state
        # self.selectiveScan = Selective_Scan(d_model=hidden, d_state=self.d_state, expand=1)

        dk, dv, num_heads = make_multihead_dims(
            hidden,
            key_ratio=key_ratio,
            head_dim=head_dim
        )



        self.gated_scan = GatedDeltaScan(
            C=hidden,
            dk=dk,
            dv=dv,
            num_heads=num_heads,
            use_prompt=use_prompt,
            use_lowrank=use_lowrank,
            rank=rank,
            use_delta=use_delta,
            chunk_size=chunk_size,
            gate=gate,
            lean=lean,
            prompt_gates=prompt_gates,
            d_state=d_state,
        )

        self.lean = lean
        if not lean:
            # project prompt d_state -> hidden
            self.prompt_proj = nn.Linear(self.d_state, hidden, bias=False)


        self.out_norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, dim, bias=True)

        self.in_proj = nn.Sequential(
            nn.Conv2d(self.dim, hidden, 1, 1, 0),
        )

        self.CPE = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),
        )

        self.embeddingB = nn.Embedding(self.num_tokens, self.inner_rank)  # [64,32] [32, 48] = [64,48]
        self.embeddingB.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)

        self.route = nn.Sequential(
            nn.Linear(self.dim, self.dim // 3),
            nn.GELU(),
            nn.Linear(self.dim // 3, self.num_tokens),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, x_size, token):
        B, n, C = x.shape
        H, W = x_size

        full_embedding = self.embeddingB.weight @ token.weight  # [128, C]

        pred_route = self.route(x)  # [B, HW, num_token]
        cls_policy = safe_gumbel_softmax(pred_route, hard=True, dim=-1)  # [B, HW, num_token]

        prompt = torch.matmul(cls_policy, full_embedding).view(B, n, self.d_state)
        if not self.lean:
            prompt = self.prompt_proj(prompt)

        detached_index = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False).view(B, n)  # [B, HW]
        x_sort_values, x_sort_indices = torch.sort(detached_index, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse(x_sort_indices)

        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.in_proj(x)
        x = x * torch.sigmoid(self.CPE(x))
        cc = x.shape[1]
        x = x.view(B, cc, -1).contiguous().permute(0, 2, 1)  

        semantic_x = semantic_neighbor(x, x_sort_indices)
        y = self.gated_scan(semantic_x, prompt)
        if self.gated_scan.last_log_alpha is not None and self.gated_scan.record_log_alpha and not self.training:
            # back to image order, as [B,H,W,n_gate] over the window-padded image
            self.gated_scan.last_log_alpha = semantic_neighbor(
                self.gated_scan.last_log_alpha, x_sort_indices_reverse).view(B, H, W, -1)
        y = self.out_proj(self.out_norm(y))
        x = semantic_neighbor(y, x_sort_indices_reverse)

        return x


class GatedDeltaScan(nn.Module):
    def __init__(self, 
        C, 
        dk=32, 
        dv=32, 
        num_heads=4, 
        use_prompt=True,
        use_lowrank=False,
        rank=16,
        use_delta=False,
        chunk_size=64,
        gate='head',
        lean=False,
        prompt_gates=('a', 'b'),
        d_state=16,
    ):
        """gate: 'head' = one decay per head; 'channel' = one decay per key channel, bounded to
        log alpha in [-1, 0] so chunk_gated_delta_rule's exponents stay finite (needs chunk_size <= 160).
        lean: size-matched variant closer to MambaIRv2's scan: v = x (no v projection), the d_state prompt
        is added to q and the gates only, no inner norm/projection (GSSM already has one), Mamba-style D skip.
        gate: 'none' = no forgetting (alpha = 1, log alpha = 0), no a_proj/ap_proj (ablation).
        prompt_gates: which of alpha ('a') and beta ('b') also receive the prompt; q always does. [] = q only.
        record_log_alpha: set True in eval to keep the last per-token log alpha in self.last_log_alpha, for the
        forgetting maps. GSSM un-sorts it to image order as [B,H,W,n_gate] (H, W = window-padded LR size).
        """
        super().__init__()
        assert dk % num_heads == 0 and dv % num_heads == 0
        assert rank > 0
        assert gate in ('head', 'channel', 'none'), gate
        assert gate != 'channel' or (not use_lowrank and chunk_size <= 160), 'channel gate: full-rank, chunk_size <= 160'
        assert set(prompt_gates) <= {'a', 'b'}, prompt_gates
        assert not lean or (dv == C and not use_lowrank), f'lean needs v = x: dv ({dv}) must equal C ({C})'

        self.C = C
        self.dk = dk
        self.dv = dv
        self.h = num_heads
        self.dk_h = dk // num_heads
        self.dv_h = dv // num_heads
        self.use_prompt = use_prompt

        self.use_lowrank = use_lowrank
        self.rank = rank
        self.use_delta = use_delta
        self.chunk_size = chunk_size
        self.gate = gate
        self.lean = lean
        self.prompt_a = use_prompt and 'a' in prompt_gates and gate != 'none'
        self.prompt_b = use_prompt and 'b' in prompt_gates
        self.record_log_alpha = False
        self.last_log_alpha = None
        n_gate = dk if gate == 'channel' else num_heads


        # projections
        self.q_proj = nn.Linear(C, dk, bias=False)
        self.k_proj = nn.Linear(C, dk, bias=False)
        if not lean:
            self.v_proj = nn.Linear(C, dv, bias=False)
        if gate != 'none':
            self.a_proj = nn.Linear(C, n_gate, bias=True)
        self.b_proj = nn.Linear(C, num_heads, bias=True)

        if use_prompt:
            p_in = d_state if lean else C
            self.qp_proj = nn.Linear(p_in, dk, bias=False)
            if not lean:
                self.kp_proj = nn.Linear(C, dk, bias=False)
                self.vp_proj = nn.Linear(C, dv, bias=False)
            if self.prompt_a:
                self.ap_proj = nn.Linear(p_in, n_gate, bias=False)
            if self.prompt_b:
                self.bp_proj = nn.Linear(p_in, num_heads, bias=False)


    
        # low-rank gate projection: g_t = G(K_t) in R^rank (per head)
        if self.use_lowrank:
            self.g_proj = nn.Linear(self.dk_h, rank, bias=False)


        if lean:
            self.D = nn.Parameter(torch.ones(C))  # skip term, as Mamba's D
        else:
            # FP32 LayerNorm (important)
            self.out_norm = nn.LayerNorm(dv)
            self.out_proj = nn.Linear(dv, C, bias=False)

            self.res_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x, prompt=None):
        """
        x: [B,L,C], prompt: [B,L,C] or None
        """
        B, L, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = x if self.lean else self.v_proj(x)
        a = self.a_proj(x) if self.gate != 'none' else None
        b = self.b_proj(x)

        if self.use_prompt:
            q = q + self.qp_proj(prompt)
            if not self.lean:
                k = k + self.kp_proj(prompt)
                v = v + self.vp_proj(prompt)
            if self.prompt_a:
                a = a + self.ap_proj(prompt)
            if self.prompt_b:
                b = b + self.bp_proj(prompt)

        # ---- split heads: [B,H,L,D] ----
        q = l2norm(q.view(B, L, self.h, self.dk_h), dim=-1).transpose(1, 2)
        k = l2norm(k.view(B, L, self.h, self.dk_h), dim=-1).transpose(1, 2)
        v = v.view(B, L, self.h, self.dv_h).transpose(1, 2)
        if self.gate == 'channel':
            log_alpha = -torch.sigmoid(a.float()).view(B, L, self.h, self.dk_h).transpose(1, 2)
        elif self.gate == 'head':
            log_alpha = F.logsigmoid(a.float()).transpose(1, 2)  # decay gate, kept in log space
        else:
            log_alpha = x.new_zeros(B, self.h, L, dtype=torch.float32)  # alpha = 1: never forget
        if self.record_log_alpha and not self.training:
            # [B,L,n_gate] in scan order; GSSM moves it back to image order
            self.last_log_alpha = (log_alpha.transpose(1, 2).reshape(B, L, -1) if log_alpha.dim() == 4
                                   else log_alpha.transpose(1, 2)).detach()
        beta = torch.sigmoid(b).transpose(1, 2)

        if not self.use_lowrank:
            o, _ = chunk_gated_delta_rule(q, k, v, log_alpha, beta, chunk_size=self.chunk_size,
                                          use_delta=self.use_delta)
        else:
            o = self._lowrank_scan(q, k, v, log_alpha.exp(), beta)
        o = o.transpose(1, 2).reshape(B, L, self.dv)
        if self.lean:
            return o.to(x.dtype) + self.D * x

        # ---- FP32 normalization + residual ----
        y = self.out_proj(self.out_norm(o.float())).to(x.dtype)
        return x + self.res_scale * y

    def _lowrank_scan(self, q, k, v, alpha, beta, T=256):
        """Legacy low-rank path, unchanged: it still has no within-chunk read (see docs/GatedMambaIR.md)."""
        B, H, L, _ = q.shape
        pad = (T - L % T) % T
        if pad:
            q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
            alpha, beta = F.pad(alpha, (0, pad), value=1.), F.pad(beta, (0, pad))
        q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
        U = q.new_zeros(B, H, self.dv_h, self.rank)
        W = q.new_zeros(B, H, self.dk_h, self.rank)
        outs = []
        for s in range(0, L + pad, T):
            Kc = k[:, :, s:s + T]
            g = self.g_proj(Kc.float()).to(torch.bfloat16)
            Oc, U, W = chunk_gated_delta_lowrank(q[:, :, s:s + T], Kc, v[:, :, s:s + T], alpha[:, :, s:s + T],
                                                 beta[:, :, s:s + T], U, W, g)
            outs.append(Oc)
        return torch.cat(outs, dim=2)[:, :, :L]


class AttentiveLayer(nn.Module):
    def __init__(self,
                 dim,
                 d_state,
                 input_resolution,
                 num_heads,
                 window_size,
                 shift_size,
                 inner_rank,
                 num_tokens,
                 convffn_kernel_size,
                 mlp_ratio,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 is_last=False,
                 use_prompt=True,
                 key_ratio=0.25,
                 head_dim=16,
                 use_lowrank=False,
                 rank=16,
                 use_delta=False,
                 chunk_size=64,
                 gate='head',
                 lean=False,
                 prompt_gates=('a', 'b'),
                 ):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.convffn_kernel_size = convffn_kernel_size
        self.num_tokens = num_tokens
        self.softmax = nn.Softmax(dim=-1)
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.is_last = is_last
        self.inner_rank = inner_rank

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)

        layer_scale = 1e-4
        self.scale1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        self.scale2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

        self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        self.win_mhsa = WindowAttention(
            self.dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        self.assm = GSSM(
            self.dim,
            d_state,
            input_resolution=input_resolution,
            num_tokens=num_tokens,
            inner_rank=inner_rank,
            mlp_ratio=mlp_ratio,
            key_ratio=key_ratio,
            head_dim=head_dim,
            use_prompt=use_prompt,
            use_lowrank=use_lowrank,
            rank=rank,
            use_delta=use_delta,
            chunk_size=chunk_size,
            gate=gate,
            lean=lean,
            prompt_gates=prompt_gates,
        )

        mlp_hidden_dim = int(dim * self.mlp_ratio)

        self.convffn1 = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size, )
        self.convffn2 = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size, )

        self.embeddingA = nn.Embedding(self.inner_rank, d_state)
        self.embeddingA.weight.data.uniform_(-1 / self.inner_rank, 1 / self.inner_rank)

    def forward(self, x, x_size, params):
        h, w = x_size
        b, n, c = x.shape
        c3 = 3 * c

        # part1: Window-MHSA
        shortcut = x
        x = self.norm1(x)
        qkv = self.wqkv(x)
        qkv = qkv.reshape(b, h, w, c3)
        if self.shift_size > 0:
            shifted_qkv = torch.roll(qkv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = params['attn_mask']
        else:
            shifted_qkv = qkv
            attn_mask = None
        x_windows = window_partition(shifted_qkv, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c3)
        attn_windows = self.win_mhsa(x_windows, rpi=params['rpi_sa'], mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        x_win = attn_x.view(b, n, c) + shortcut
        x_win = self.convffn1(self.norm2(x_win), x_size) + x_win
        x = shortcut * self.scale1 + x_win

        # part2: Attentive State Space
        shortcut = x
        x_aca = self.assm(self.norm3(x), x_size, self.embeddingA) + x
        x = x_aca + self.convffn2(self.norm4(x_aca), x_size)
        x = shortcut * self.scale2 + x

        return x


class BasicBlock(nn.Module):
    """ A basic ASSB for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        idx (int): Block index.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        num_tokens (int): Token number for each token dictionary.
        convffn_kernel_size (int): Convolutional kernel size for ConvFFN.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 d_state,
                 input_resolution,
                 idx,
                 depth,
                 num_heads,
                 window_size,
                 inner_rank,
                 num_tokens,
                 convffn_kernel_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 key_ratio=0.25,
                 head_dim=16,
                 downsample=None, 
                 use_checkpoint=False, 
                 use_prompt=True,
                 use_lowrank=False,
                 rank=16,
                 use_delta=False,
                 chunk_size=64,
                 gate='head',
                 lean=False,
                 prompt_gates=('a', 'b'),
            ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.idx = idx

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                AttentiveLayer(
                    dim=dim,
                    d_state=d_state,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    inner_rank=inner_rank,
                    num_tokens=num_tokens,
                    convffn_kernel_size=convffn_kernel_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    is_last=i == depth - 1,
                    key_ratio=key_ratio,
                    head_dim=head_dim,
                    use_prompt=use_prompt,
                    use_lowrank=use_lowrank,
                    rank=rank,
                    use_delta=use_delta,
                    chunk_size=chunk_size,
                    gate=gate,
                    lean=lean,
                    prompt_gates=prompt_gates,
                )
            )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size, params):
        b, n, c = x.shape
        for layer in self.layers:
            x = layer(x, x_size, params)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'


class ASSB(nn.Module):
    def __init__(self,
                 dim,
                 d_state,
                 idx,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 inner_rank,
                 num_tokens,
                 convffn_kernel_size,
                 mlp_ratio,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 key_ratio=0.25,
                 head_dim=16,
                 resi_connection='1conv', 
                 use_prompt=True,
                 use_lowrank=False,
                 rank=16,
                 use_delta=False,
                 chunk_size=64,
                 gate='head',
                 lean=False,
                 prompt_gates=('a', 'b'),
            ):
        super(ASSB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.residual_group = BasicBlock(
            dim=dim,
            d_state=d_state,
            input_resolution=input_resolution,
            idx=idx,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            num_tokens=num_tokens,
            inner_rank=inner_rank,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            downsample=downsample,
            key_ratio=key_ratio,
            head_dim=head_dim,
            use_checkpoint=use_checkpoint,
            use_prompt=use_prompt,
            use_lowrank=use_lowrank,
            rank=rank,
            use_delta=use_delta,
            chunk_size=chunk_size,
            gate=gate,
            lean=lean,
            prompt_gates=prompt_gates,
        )

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

    def forward(self, x, x_size, params):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size, params), x_size))) + x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, input_resolution=None):
        flops = 0
        h, w = self.img_size if input_resolution is None else input_resolution
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

    def flops(self, input_resolution=None):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        self.scale = scale
        self.num_feat = num_feat
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

    def flops(self, input_resolution):
        flops = 0
        x, y = input_resolution
        if (self.scale & (self.scale - 1)) == 0:
            flops += self.num_feat * 4 * self.num_feat * 9 * x * y * int(math.log(self.scale, 2))
        else:
            flops += self.num_feat * 9 * self.num_feat * 9 * x * y
        return flops


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self, input_resolution):
        flops = 0
        h, w = self.patches_resolution if input_resolution is None else input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops


@ARCH_REGISTRY.register()
class GatedMambaIR(nn.Module):
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=48,
                 d_state=8,
                 depths=(6, 6, 6, 6,),
                 num_heads=(4, 4, 4, 4,),
                 window_size=16,
                 inner_rank=32,
                 num_tokens=64,
                 convffn_kernel_size=5,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 key_ratio=0.25,
                 head_dim=16,
                 use_prompt=True,
                 use_lowrank=False,
                 rank=16,
                 use_delta=False,
                 chunk_size=64,
                 gate='head',
                 lean=False,
                 prompt_gates=('a', 'b'),
                 **kwargs):
        super().__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # relative position index
        relative_position_index_SA = self.calculate_rpi_sa()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ASSB(
                dim=embed_dim,
                d_state=d_state,
                idx=i_layer,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                inner_rank=inner_rank,
                num_tokens=num_tokens,
                convffn_kernel_size=convffn_kernel_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                key_ratio=key_ratio,
                head_dim=head_dim,
                use_prompt=use_prompt,
                use_lowrank=use_lowrank,
                rank=rank,
                use_delta=use_delta,
                chunk_size=chunk_size,
                gate=gate,
                lean=lean,
                prompt_gates=prompt_gates,
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, params):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed

        for layer in self.layers:
            x = layer(x, x_size, params)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def calculate_rpi_sa(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index

    def calculate_mask(self, x_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x):
        # padding
        h_ori, w_ori = x.size()[-2], x.size()[-1]
        mod = self.window_size
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        h, w = h_ori + h_pad, w_ori + w_pad
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h, :]
        x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w]

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        attn_mask = self.calculate_mask([h, w]).to(x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first, params)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        # unpadding
        x = x[..., :h_ori * self.upscale, :w_ori * self.upscale]

        return x



if __name__ == '__main__':
    upscale = 4
    model = MambaIRv2(
        upscale=2,
        img_size=64,
        embed_dim=48,
        d_state=8,
        depths=[5, 5, 5, 5],
        num_heads=[4, 4, 4, 4],
        window_size=16,
        inner_rank=32,
        num_tokens=64,
        convffn_kernel_size=5,
        img_range=1.,
        mlp_ratio=1.,
        upsampler='pixelshuffledirect').cuda()

    # Model Size
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.3fM" % (total / 1e6))
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(trainable_num)

    # Test
    _input = torch.randn([2, 3, 64, 64]).cuda()
    output = model(_input).cuda()
    print(output.shape)

