"""Fig. B of the DeltaSR ablations (docs/ABLATIONS.md): the learned forgetting map.

For each LR image: input | per-pixel mean forget strength of ours | the same for Table 3 exp1 (prompt only in q).
Forget strength = -log alpha (= sigmoid(a) for the channel gate), averaged over the memory entries (key
channels), at one GSSM layer. The scan runs over the semantically sorted tokens; GatedDeltaScan.record_log_alpha
gives the map back in image order (GSSM un-sorts it), as analysis/memory_heatmap.py does for its heatmaps.

Usage (from the repo root, on a GPU node):

    python analysis/forget_map.py --images datasets/SR/Urban100/LR_bicubic/X2/img004x2.png \
        datasets/SR/Urban100/LR_bicubic/X2/img012x2.png --layer 2.3

A model whose checkpoint does not exist yet (e.g. the prompt-only-in-q run while it trains) is skipped; rerun
when it has finished. Both maps of an image share one colour scale.

Outputs in --out_dir (default results/forget_map):
    <image>_L<g.l>.png                    side by side: input | model panels, with a shared colour bar
    <image>_L<g.l>_{input,<key>}.png      single panels without axes, for composing the figure in LaTeX
    maps.npz                              raw maps, for re-plotting
"""
import argparse
import os
import sys
from os import path as osp

import cv2
import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir))
sys.path.insert(0, ROOT)
from analysis.memory_heatmap import build_model, get_ssm, route_mode  # noqa: E402

MODELS = [  # key, label, test config
    ('ours', 'ours (P in q, alpha, beta)',
     'options/test/gatedmambair/test_GatedMambaIR_SR_x2_lean_chgate_50k.yml'),
    ('qonly', 'P only in q (Table 3 exp1)',
     'options/test/gatedmambair/test_GatedMambaIR_SR_x2_abl_promptq_50k.yml'),
]


def forget_strength(net, layer, img):
    """[h, w] mean over memory entries of -log alpha at GSSM `layer` ('group.block'), for an LR image [1,3,h,w]."""
    group, block = (int(v) for v in layer.split('.'))
    scan = get_ssm(net, group, block).gated_scan
    scan.record_log_alpha, scan.last_log_alpha = True, None
    try:
        with torch.no_grad():
            net(img)
        la = scan.last_log_alpha  # [1, H, W, n_gate] over the window-padded image
    finally:
        scan.record_log_alpha, scan.last_log_alpha = False, None
    h, w = img.shape[-2:]
    return (-la[0, :h, :w]).mean(-1).float().cpu().numpy()


def save_panel(path, arr, cmap=None, vmin=None, vmax=None):
    plt.imsave(path, arr, cmap=cmap, vmin=vmin, vmax=vmax)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--images', nargs='+', required=True, help='LR images (e.g. Urban100 LR_bicubic/X2)')
    p.add_argument('--layer', default='2.3', help='"group.block" of the GSSM to read (0-based)')
    p.add_argument('--crop', default=None, help='"row,col,height,width" crop of every LR image')
    p.add_argument('--route', choices=['argmax', 'gumbel'], default='argmax',
                   help='router: noise-free argmax (default) or the Gumbel sample used in training/testing')
    p.add_argument('--pct', type=float, nargs=2, default=[1, 99], help='colour scale: shared percentiles')
    p.add_argument('--cmap', default='magma')
    p.add_argument('--out_dir', default='results/forget_map')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    out = osp.join(ROOT, args.out_dir)
    os.makedirs(out, exist_ok=True)

    models = []
    for key, label, opt in MODELS:
        with open(osp.join(ROOT, opt)) as f:
            ckpt = yaml.safe_load(f)['path']['pretrain_network_g']
        if not osp.exists(ckpt):
            print(f'SKIP {key}: {ckpt} does not exist yet')
            continue
        models.append((key, label, build_model(opt, device=args.device)))
    assert models, 'no model checkpoint found'

    saved = {}
    with route_mode(args.route):
        for path in args.images:
            img = cv2.imread(osp.join(ROOT, path), cv2.IMREAD_COLOR)
            assert img is not None, f'cannot read {path}'
            img = img[:, :, ::-1]
            if args.crop:
                r, c, hh, ww = (int(v) for v in args.crop.split(','))
                img = img[r:r + hh, c:c + ww]
            x = torch.from_numpy(img.astype(np.float32).copy() / 255.).permute(2, 0, 1)[None].to(args.device)
            maps = {key: forget_strength(net, args.layer, x) for key, _, net in models}
            stem = f"{osp.splitext(osp.basename(path))[0]}_L{args.layer}"
            vmin, vmax = np.percentile(np.concatenate([m.ravel() for m in maps.values()]), args.pct)

            save_panel(osp.join(out, f'{stem}_input.png'), img)
            for key, m in maps.items():
                save_panel(osp.join(out, f'{stem}_{key}.png'), m, args.cmap, vmin, vmax)
                saved[f'{stem}_{key}'] = m
                print(f'{stem} {key}: forget strength mean {m.mean():.3f}, std {m.std():.3f}, '
                      f'range {m.min():.3f}..{m.max():.3f}', flush=True)

            n = 1 + len(models)
            fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2 * img.shape[0] / img.shape[1] + 0.6))
            axes[0].imshow(img)
            axes[0].set_title('input (LR)')
            for ax, (key, label, _) in zip(axes[1:], models):
                im = ax.imshow(maps[key], cmap=args.cmap, vmin=vmin, vmax=vmax)
                ax.set_title(f'{label}\nmean {maps[key].mean():.3f}', fontsize=10)
            for ax in axes:
                ax.axis('off')
            fig.colorbar(im, ax=axes[1:].tolist(), fraction=0.025, pad=0.01, label='forget strength  -log alpha')
            fig.suptitle(f'{osp.basename(path)}, GSSM layer {args.layer}', fontsize=10)
            fig.savefig(osp.join(out, f'{stem}.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
    np.savez_compressed(osp.join(out, 'maps.npz'), **saved)
    print('wrote', out)


if __name__ == '__main__':
    main()
