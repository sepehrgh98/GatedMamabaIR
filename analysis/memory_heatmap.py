"""Motivation figure: where does one pixel read its memory from?

For a query pixel q and one state-space layer, we measure how much every input token x_i of that layer's
scan contributes to the scan output o_q:  heat_i = ||d o_q / d x_i||_F  (Jacobian block norm, estimated with
random probe vectors), in the spirit of "The Hidden Attention of Mamba Models" (Ali et al., 2024).
The scan runs over the semantically sorted sequence, so the tokens are un-sorted back to image positions
and drawn as a heatmap over the LR image. MambaIRv2 (ASSM / Selective_Scan) and GatedMambaIR (GSSM /
GatedDeltaScan) are probed at the same layer position.

Usage (from the repo root, on a GPU node):

    python analysis/memory_heatmap.py \
        --images datasets/SR/Urban100/LR_bicubic/X2/img_004x2.png \
        --query 120,200 --query 60,310 --layers 2.3 3.3

The two models are built from their test configs (network_g + path.pretrain_network_g); the defaults are the
x2 500k runs. Use --ckpt_a / --ckpt_b to point at other checkpoints.
--query is "row,col" in pixels of the image actually fed to the model (after --crop). Every run also saves
<image>_grid.png with pixel ticks, to pick query points on windows. Without --query the image centre is used.

Outputs in --out_dir (default results/memory_heatmap):
    <image>_grid.png                      input with pixel ticks
    <image>_q<r>_<c>_L<g.l>.png           side by side: input + query | model A | model B
    <image>_q<r>_<c>_L<g.l>_{input,a,b}.png   single panels without axes, for composing the figure in LaTeX
    <image>_q<r>_<c>_mean.png             same, averaged over the probed layers (if more than one)
    heat.npz                              raw maps, for re-plotting
    stats.csv                             concentration numbers per (image, query, layer, model)

Note: the scan is causal over the sorted sequence, so q can only read tokens sorted before it (heat is 0
elsewhere). stats.csv reports how many tokens precede q and how many share its semantic group.
"""
import argparse
import csv
import os
import sys
from os import path as osp

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import yaml

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir))
sys.path.insert(0, ROOT)
from basicsr.archs import gatedmambair_arch, mambairv2_arch  # noqa: E402
from basicsr.archs.gatedmambair_arch import GatedMambaIR  # noqa: E402
from basicsr.archs.mambairv2_arch import MambaIRv2  # noqa: E402

ARCHS = {'MambaIRv2': MambaIRv2, 'GatedMambaIR': GatedMambaIR}
DEFAULT_A = 'options/test/mambairv2/test_MambaIRv2_SR_x2_500k.yml'
DEFAULT_B = 'options/test/gatedmambair/test_GatedMambaIR_SR_x2_lean_chgate_500k.yml'


# ----------------------------------------------------------------------------- models

def build_model(opt_path, ckpt=None, device='cuda'):
    with open(osp.join(ROOT, opt_path) if not osp.isabs(opt_path) else opt_path) as f:
        opt = yaml.safe_load(f)
    net_opt = dict(opt['network_g'])
    net = ARCHS[net_opt.pop('type')](**net_opt)
    ckpt = ckpt or opt['path']['pretrain_network_g']
    state = torch.load(ckpt, map_location='cpu')
    key = 'params_ema' if 'params_ema' in state else 'params' if 'params' in state else None
    state = state[key] if key else state
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    net.load_state_dict(state, strict=True)
    print(f'loaded {net.__class__.__name__} from {ckpt} [{key}]')
    return net.to(device).eval()


def get_ssm(net, group, layer):
    return net.layers[group].residual_group.layers[layer].assm


def argmax_onehot(logits, tau=1.0, hard=True, dim=-1):
    """Noise-free routing: the hard one-hot of the router's argmax (instead of a Gumbel sample)."""
    return torch.zeros_like(logits).scatter_(dim, logits.argmax(dim, keepdim=True), 1.0)


class route_mode:
    """Context: replace Gumbel sampling in both archs by argmax (route='argmax'), or keep it (route='gumbel')."""

    def __init__(self, route):
        self.route = route

    def __enter__(self):
        self.saved = (F.gumbel_softmax, gatedmambair_arch.safe_gumbel_softmax)
        if self.route == 'argmax':
            F.gumbel_softmax = argmax_onehot  # used by mambairv2_arch as F.gumbel_softmax
            gatedmambair_arch.safe_gumbel_softmax = argmax_onehot

    def __exit__(self, *exc):
        F.gumbel_softmax, gatedmambair_arch.safe_gumbel_softmax = self.saved


# ----------------------------------------------------------------------------- probe

def scan_of(ssm):
    return ssm.gated_scan if hasattr(ssm, 'gated_scan') else ssm.selectiveScan


def make_probed_forward(ssm, queries, n_probe, jac, record):
    """ASSM.forward / GSSM.forward (same code in both archs), plus the Jacobian probe around the scan.

    queries: flat token indices (row * W + col on the padded grid). Fills record with
    heat [Q, HW] (image order), group [HW] (route index per token), rank [HW] (position in the scan).
    """
    def forward(x, x_size, token):
        B, n, C = x.shape
        H, W = x_size
        full_embedding = ssm.embeddingB.weight @ token.weight
        pred_route = ssm.route(x)
        # looked up at call time, so route_mode's patch applies
        route_fn = gatedmambair_arch.safe_gumbel_softmax if hasattr(ssm, 'gated_scan') else F.gumbel_softmax
        cls_policy = route_fn(pred_route, hard=True, dim=-1)
        prompt = torch.matmul(cls_policy, full_embedding).view(B, n, ssm.d_state)
        if not getattr(ssm, 'lean', True):  # non-lean GSSM projects the prompt; ASSM has no lean attr
            prompt = ssm.prompt_proj(prompt)

        detached_index = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False).view(B, n)
        _, x_sort_indices = torch.sort(detached_index, dim=-1, stable=False)
        x_sort_indices_reverse = mambairv2_arch.index_reverse(x_sort_indices)

        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = ssm.in_proj(x)
        x = x * torch.sigmoid(ssm.CPE(x))
        cc = x.shape[1]
        x = x.view(B, cc, -1).contiguous().permute(0, 2, 1)
        semantic_x = mambairv2_arch.semantic_neighbor(x, x_sort_indices)

        # ---- probe: d o_q / d x_i through the scan only (routing, order and prompt held fixed) ----
        assert B == 1
        rev = x_sort_indices_reverse[0]  # image index -> scan position
        heats = []
        with torch.enable_grad():
            sx = semantic_x.detach().float().requires_grad_(True)
            y_probe = scan_of(ssm)(sx, prompt.detach().float())  # [1, L, C]
            for qi, q in enumerate(queries):
                o_q = y_probe[0, rev[q]]
                if jac == 'sum':
                    g, = torch.autograd.grad(o_q.sum(), sx, retain_graph=True)
                    h = g[0].abs().sum(-1)
                else:  # Frobenius norm of each Jacobian block, Hutchinson estimate
                    gen = torch.Generator(device=o_q.device).manual_seed(qi)
                    h = torch.zeros(sx.shape[1], device=sx.device)
                    for _ in range(n_probe):
                        r = torch.randn(o_q.shape, device=o_q.device, generator=gen)
                        g, = torch.autograd.grad((o_q * r).sum(), sx, retain_graph=True)
                        h += g[0].pow(2).sum(-1)
                    h = (h / n_probe).sqrt()
                heats.append(h[rev].detach().cpu())  # back to image order
            del y_probe, sx
        record['heat'] = torch.stack(heats).numpy()
        record['group'] = detached_index[0].cpu().numpy()
        record['rank'] = rev.cpu().numpy()

        y = scan_of(ssm)(semantic_x, prompt)
        y = ssm.out_proj(ssm.out_norm(y))
        return mambairv2_arch.semantic_neighbor(y, x_sort_indices_reverse)

    return forward


@torch.no_grad()
def probe_model(net, img, layers, queries_rc, n_probe, jac, route, seed):
    """img: [1,3,h,w] in [0,1]. Returns {layer: record} with maps cropped to h x w."""
    h0, w0 = img.shape[-2:]
    ws = net.window_size
    W = ((w0 + ws - 1) // ws) * ws  # padded width used inside the model
    queries = [r * W + c for r, c in queries_rc]
    records = {}
    for g, l in layers:
        ssm = get_ssm(net, g, l)
        records[(g, l)] = {}
        ssm.forward = make_probed_forward(ssm, queries, n_probe, jac, records[(g, l)])
    torch.manual_seed(seed)
    try:
        with route_mode(route):
            net(img)
    finally:
        for g, l in layers:
            del get_ssm(net, g, l).forward  # back to the class method
    H = records[layers[0]]['group'].size // W
    for rec in records.values():
        rec['heat'] = rec['heat'].reshape(len(queries), H, W)[:, :h0, :w0]
        rec['group'] = rec['group'].reshape(H, W)[:h0, :w0]
        rec['rank'] = rec['rank'].reshape(H, W)[:h0, :w0]
    return records


# ----------------------------------------------------------------------------- stats and plots

def concentration(heat, q, group, rank, local):
    """Numbers for one map. heat [h,w]; the query token itself (Mamba-style D skip) is excluded."""
    r, c = q
    hm = heat.copy()
    hm[r, c] = 0
    p = hm.ravel() / max(hm.sum(), 1e-12)
    nz = p[p > 0]
    yy, xx = np.mgrid[:hm.shape[0], :hm.shape[1]]
    far = (np.abs(yy - r) > local) | (np.abs(xx - c) > local)
    same = group == group[r, c]
    k = max(1, int(0.01 * p.size))
    return {
        'eff_tokens': float(np.exp(-(nz * np.log(nz)).sum())),  # exp(entropy): how many tokens it reads
        'top1pct_mass': float(np.sort(p)[-k:].sum()),
        'far_mass': float(p.reshape(hm.shape)[far].sum()),  # mass outside the +-local window around q
        'same_group_mass': float(p.reshape(hm.shape)[same].sum()),
        'tokens_before_q': int((rank < rank[r, c]).sum()),
        'group_size': int(same.sum()),
        'group_before_q': int((same & (rank < rank[r, c])).sum()),
    }


def display_map(heat, q, pct, gamma):
    hm = heat.copy()
    hm[q[0], q[1]] = 0
    vmax = np.percentile(hm[hm > 0], pct) if (hm > 0).any() else 1.0
    return np.clip(hm / max(vmax, 1e-12), 0, 1) ** gamma


def overlay(ax, gray, hm, q, alpha, cmap):
    ax.imshow(gray, cmap='gray', vmin=0, vmax=1)
    ax.imshow(hm, cmap=cmap, alpha=alpha * hm ** 0.5, vmin=0, vmax=1)  # weak values fade into the image
    ax.plot(q[1], q[0], marker='*', ms=14, mec='white', mfc='lime', mew=1.2)
    ax.set_axis_off()


def save_panel(path, gray, hm, q, alpha, cmap, rgb=None):
    h, w = gray.shape
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    if rgb is not None:
        ax.imshow(rgb)
        ax.plot(q[1], q[0], marker='*', ms=14, mec='white', mfc='lime', mew=1.2)
        ax.set_axis_off()
    else:
        overlay(ax, gray, hm, q, alpha, cmap)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_figure(path, rgb, gray, maps, labels, q, alpha, cmap, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5 * gray.shape[0] / gray.shape[1] + 0.6))
    axes[0].imshow(rgb)
    axes[0].plot(q[1], q[0], marker='*', ms=14, mec='white', mfc='lime', mew=1.2)
    axes[0].set_title('LR input, query *')
    axes[0].set_axis_off()
    for ax, hm, lab in zip(axes[1:], maps, labels):
        overlay(ax, gray, hm, q, alpha, cmap)
        ax.set_title(lab)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_grid(path, rgb, step):
    h, w = rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(max(6, w / 50), max(4, h / 50)))
    ax.imshow(rgb)
    ax.set_xticks(np.arange(0, w, step))
    ax.set_yticks(np.arange(0, h, step))
    ax.tick_params(labelsize=6)
    ax.grid(color='yellow', lw=0.3, alpha=0.6)
    ax.set_xlabel('col')
    ax.set_ylabel('row')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--images', nargs='+', required=True, help='LR images (e.g. Urban100 LR_bicubic/X2)')
    p.add_argument('--query', action='append', default=[], help='"row,col" in the fed image; repeatable')
    p.add_argument('--layers', nargs='+', default=['2.3', '3.3'],
                   help='"group.layer" (0-based; 6 groups x 6 layers in the base models)')
    p.add_argument('--opt_a', default=DEFAULT_A, help='test yml of model A (network_g + pretrain path)')
    p.add_argument('--opt_b', default=DEFAULT_B, help='test yml of model B')
    p.add_argument('--ckpt_a', default=None, help='override checkpoint of model A')
    p.add_argument('--ckpt_b', default=None, help='override checkpoint of model B')
    p.add_argument('--label_a', default='MambaIRv2')
    p.add_argument('--label_b', default='DeltaSR (ours)')
    p.add_argument('--crop', default=None, help='"row,col,height,width" crop of the LR image before probing')
    p.add_argument('--jac', choices=['frob', 'sum'], default='frob',
                   help='frob: ||J_qi||_F via random probes; sum: |grad of sum_c o_q[c]| summed over channels')
    p.add_argument('--n_probe', type=int, default=16, help='random probes per query for --jac frob')
    p.add_argument('--route', choices=['argmax', 'gumbel'], default='argmax',
                   help='argmax: deterministic routing; gumbel: as in the test script (seeded)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--local', type=int, default=8, help='half size of the "local" window for far_mass')
    p.add_argument('--pct', type=float, default=99.5, help='display: normalise each map by this percentile')
    p.add_argument('--gamma', type=float, default=0.5, help='display: power applied after normalising')
    p.add_argument('--alpha', type=float, default=0.85, help='display: max heatmap opacity')
    p.add_argument('--cmap', default='jet')
    p.add_argument('--grid_step', type=int, default=20)
    p.add_argument('--out_dir', default='results/memory_heatmap')
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    layers = [tuple(int(v) for v in s.split('.')) for s in args.layers]
    nets = [build_model(args.opt_a, args.ckpt_a, args.device), build_model(args.opt_b, args.ckpt_b, args.device)]
    labels = [args.label_a, args.label_b]

    rows, arrays = [], {}
    for img_path in args.images:
        name = osp.splitext(osp.basename(img_path))[0]
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        assert bgr is not None, f'cannot read {img_path}'
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        if args.crop:
            r0, c0, ch, cw = (int(v) for v in args.crop.split(','))
            rgb = rgb[r0:r0 + ch, c0:c0 + cw]
            name += f'_crop{r0}_{c0}'
        h, w = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        save_grid(osp.join(args.out_dir, f'{name}_grid.png'), rgb, args.grid_step)

        queries = [tuple(int(v) for v in s.split(',')) for s in args.query] or [(h // 2, w // 2)]
        for r, c in queries:
            assert 0 <= r < h and 0 <= c < w, f'query {(r, c)} outside image {h}x{w}'
        img = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(args.device)

        recs = [probe_model(net, img, layers, queries, args.n_probe, args.jac, args.route, args.seed)
                for net in nets]
        print(f'{name}: {h}x{w}, {len(queries)} queries, layers {args.layers}')

        for qi, q in enumerate(queries):
            qtag = f'{name}_q{q[0]}_{q[1]}'
            means = [np.zeros((h, w)) for _ in nets]
            for g, l in layers:
                ltag = f'{qtag}_L{g}.{l}'
                heats = []
                for m, (rec, lab) in enumerate(zip(recs, labels)):
                    heat = rec[(g, l)]['heat'][qi]
                    heats.append(heat)
                    hn = heat.copy()
                    hn[q] = 0
                    means[m] += hn / max(hn.sum(), 1e-12)
                    arrays[f'{ltag}_{"ab"[m]}'] = heat
                    st = concentration(heat, q, rec[(g, l)]['group'], rec[(g, l)]['rank'], args.local)
                    rows.append({'image': name, 'query': f'{q[0]},{q[1]}', 'layer': f'{g}.{l}', 'model': lab, **st})
                maps = [display_map(hm, q, args.pct, args.gamma) for hm in heats]
                save_figure(osp.join(args.out_dir, f'{ltag}.png'), rgb, gray, maps, labels, q, args.alpha,
                            args.cmap, f'{name}  query (row,col)={q}  layer {g}.{l}  |d o_q / d x_i|')
                save_panel(osp.join(args.out_dir, f'{ltag}_input.png'), gray, None, q, args.alpha, args.cmap, rgb)
                for m, hm in enumerate(maps):
                    save_panel(osp.join(args.out_dir, f'{ltag}_{"ab"[m]}.png'), gray, hm, q, args.alpha, args.cmap)
            if len(layers) > 1:
                maps = [display_map(mn, q, args.pct, args.gamma) for mn in means]
                save_figure(osp.join(args.out_dir, f'{qtag}_mean.png'), rgb, gray, maps, labels, q, args.alpha,
                            args.cmap, f'{name}  query (row,col)={q}  mean over layers {" ".join(args.layers)}')
                for m, hm in enumerate(maps):
                    save_panel(osp.join(args.out_dir, f'{qtag}_mean_{"ab"[m]}.png'), gray, hm, q, args.alpha,
                               args.cmap)
        torch.cuda.empty_cache()

    np.savez_compressed(osp.join(args.out_dir, 'heat.npz'), **arrays)
    with open(osp.join(args.out_dir, 'stats.csv'), 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f'{"image":<22}{"query":>10}{"layer":>7}  {"model":<16}{"eff_tok":>9}{"top1%":>7}{"far":>6}{"same_grp":>9}')
    for r in rows:
        print(f'{r["image"]:<22}{r["query"]:>10}{r["layer"]:>7}  {r["model"]:<16}{r["eff_tokens"]:>9.0f}'
              f'{r["top1pct_mass"]:>7.2f}{r["far_mass"]:>6.2f}{r["same_group_mass"]:>9.2f}')
    print(f'saved to {args.out_dir}')


if __name__ == '__main__':
    main()
