"""Fig. 4 (x4 qualitative comparison): shortlist, best crop per image, preview panels and raw crops.

Reads docs/fig4_qualitative/per_image_x4.csv (analysis/per_image_metrics.py) and the SR images saved by each model's x4 test.
1. Rank images by gain = min(DeltaSR - HAT, DeltaSR - MambaIRv2) in full-image PSNR; keep the top 10 Urban100 + top 5 Manga109.
2. For each, slide a 160x160 HR window (= 40x40 LR, stride 16) over the image (inside crop_border 4) and take the window
   with the largest local gain min(DeltaSR - HAT, DeltaSR - MambaIRv2) in Y-PSNR, skipping flat windows (HR Y std < 10).
3. Write shortlist.md, preview/<dataset>_<image>.png (HR with the box, 2x4 crops: HR + figure models, nearest-neighbour x2,
   model name and crop PSNR; grey "pending" tile for models without x4 outputs yet) and crops/<dataset>_<image>/<model>.png
   (raw 160x160 crops, lossless).
Both rankings use DeltaSR, HAT and MambaIRv2 only, so re-running after DAT / CAT-A / MambaIR finish keeps the same images and
crops and only fills in their tiles.

usage (on a compute node, from the repo root or a worktree):
  python analysis/fig4_qualitative.py --root /speed-scratch/se_gham/GatedMambaIR
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from basicsr.metrics.metric_util import to_y_channel  # noqa: E402
from per_image_metrics import MODELS, LQ_TMPL, SETS  # noqa: E402

BORDER = 4
WIN, STRIDE, MIN_STD = 160, 16, 10.0
TOP = {'Urban100': 10, 'Manga109': 5}
FIGURE = ['HR', 'SwinIR', 'HAT', 'DAT', 'CAT-A', 'MambaIR', 'MambaIRv2', 'DeltaSR']
TABLE = ['DeltaSR', 'MambaIRv2', 'HAT', 'DAT', 'CAT-A', 'MambaIR', 'SwinIR', 'RCAN', 'EDSR', 'SAN']
SUFFIX = '_x4ft250k'
TILE, LABEL_H = WIN * 2, 44


def sr_path(root, model, dataset, image):
    p = os.path.join(root, 'results', MODELS[model], 'visualization', dataset,
                     LQ_TMPL[dataset].format(image) + SUFFIX + '.png')
    return p if os.path.isfile(p) else None


def y_of(img):
    # same Y conversion as calculate_psnr(test_y_channel=True): (H, W, 1) on 0-255
    return to_y_channel(img)[..., 0].astype(np.float64)


def window_sums(a):
    """Sums of a over all WIN x WIN windows at STRIDE (integral image)."""
    ii = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    h, w = a.shape
    ys = np.arange(0, h - WIN + 1, STRIDE)
    xs = np.arange(0, w - WIN + 1, STRIDE)
    Y, X = np.meshgrid(ys, xs, indexing='ij')
    s = ii[Y + WIN, X + WIN] - ii[Y, X + WIN] - ii[Y + WIN, X] + ii[Y, X]
    return s, ys, xs


def local_psnr(sr_y, hr_y):
    s, _, _ = window_sums((sr_y - hr_y) ** 2)
    mse = np.maximum(s / WIN**2, 1e-10)
    return 10 * np.log10(255.0**2 / mse)


def crop_psnr(sr_y, hr_y):
    mse = max(np.mean((sr_y - hr_y) ** 2), 1e-10)
    return 10 * np.log10(255.0**2 / mse)


def best_crop(root, dataset, image):
    hr = cv2.imread(os.path.join(root, 'datasets/SR', dataset, 'HR', image + '.png'), cv2.IMREAD_COLOR)
    imgs = {m: cv2.imread(sr_path(root, m, dataset, image), cv2.IMREAD_COLOR) for m in ('DeltaSR', 'HAT', 'MambaIRv2')}
    h, w = imgs['DeltaSR'].shape[:2]
    hr = hr[:h, :w]
    b = BORDER
    hr_y = y_of(hr)[b:-b, b:-b]
    p = {m: local_psnr(y_of(im)[b:-b, b:-b], hr_y) for m, im in imgs.items()}
    gain = np.minimum(p['DeltaSR'] - p['HAT'], p['DeltaSR'] - p['MambaIRv2'])
    s1, ys, xs = window_sums(hr_y)
    s2, _, _ = window_sums(hr_y**2)
    std = np.sqrt(np.maximum(s2 / WIN**2 - (s1 / WIN**2) ** 2, 0))
    gain = np.where(std >= MIN_STD, gain, -np.inf)
    r, c = np.unravel_index(np.argmax(gain), gain.shape)
    return dict(x=int(xs[c] + b), y=int(ys[r] + b), w=WIN, h=WIN, crop_gain=float(gain[r, c]), hr_std=float(std[r, c]))


def font(size):
    for f in ('DejaVuSans.ttf', '/usr/share/fonts/dejavu/DejaVuSans.ttf', '/usr/share/fonts/TTF/DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def preview(root, out_dir, dataset, image, box):
    hr = cv2.imread(os.path.join(root, 'datasets/SR', dataset, 'HR', image + '.png'), cv2.IMREAD_COLOR)
    x, y, w, h = box['x'], box['y'], box['w'], box['h']
    crop_dir = os.path.join(out_dir, 'crops', f'{dataset}_{image}')
    os.makedirs(crop_dir, exist_ok=True)
    f = font(22)
    tiles = []
    hr_crop = hr[y:y + h, x:x + w]
    hr_y = y_of(hr_crop)
    psnrs = {}
    for m in FIGURE:
        if m == 'HR':
            img, label = hr_crop, 'HR'
        else:
            pth = sr_path(root, m, dataset, image)
            if pth is None:
                img, label = None, f'{m} (pending)'
            else:
                sr = cv2.imread(pth, cv2.IMREAD_COLOR)
                img = sr[y:y + h, x:x + w]
                psnrs[m] = crop_psnr(y_of(img), hr_y)
                label = f'{m}  {psnrs[m]:.2f} dB'
        tile = Image.new('RGB', (TILE, TILE + LABEL_H), 'white')
        if img is None:
            tile.paste(Image.new('RGB', (TILE, TILE), (170, 170, 170)), (0, 0))
            ImageDraw.Draw(tile).text((TILE // 2, TILE // 2), 'pending', fill='white', font=f, anchor='mm')
        else:
            cv2.imwrite(os.path.join(crop_dir, f'{m}.png'), img)
            rgb = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize((TILE, TILE), Image.NEAREST)
            tile.paste(rgb, (0, 0))
        bold = m == 'DeltaSR'
        ImageDraw.Draw(tile).text((TILE // 2, TILE + LABEL_H // 2), label, fill=(200, 0, 0) if bold else 'black',
                                  font=f, anchor='mm')
        tiles.append(tile)

    right = Image.new('RGB', (4 * TILE, 2 * (TILE + LABEL_H)), 'white')
    for i, t in enumerate(tiles):
        right.paste(t, ((i % 4) * TILE, (i // 4) * (TILE + LABEL_H)))
    full = Image.fromarray(cv2.cvtColor(hr, cv2.COLOR_BGR2RGB))
    s = right.height / full.height
    full = full.resize((max(1, round(full.width * s)), right.height), Image.LANCZOS)
    ImageDraw.Draw(full).rectangle([x * s, y * s, (x + w) * s, (y + h) * s], outline=(255, 0, 0), width=3)
    canvas = Image.new('RGB', (full.width + 16 + right.width, right.height), 'white')
    canvas.paste(full, (0, 0))
    canvas.paste(right, (full.width + 16, 0))
    os.makedirs(os.path.join(out_dir, 'preview'), exist_ok=True)
    out = os.path.join(out_dir, 'preview', f'{dataset}_{image}.png')
    canvas.save(out)
    return out, psnrs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/speed-scratch/se_gham/GatedMambaIR')
    p.add_argument('--out', default='docs/fig4_qualitative')
    args = p.parse_args()

    per = {}
    for r in csv.DictReader(open(os.path.join(args.out, 'per_image_x4.csv'))):
        per[(r['dataset'], r['image'], r['model'])] = float(r['psnr'])

    rows = []
    for dataset in SETS:
        images = sorted({i for d, i, m in per if d == dataset and m == 'DeltaSR'})
        ranked = []
        for i in images:
            d = per[(dataset, i, 'DeltaSR')]
            g = min(d - per[(dataset, i, 'HAT')], d - per[(dataset, i, 'MambaIRv2')])
            ranked.append((g, i))
        ranked.sort(reverse=True)
        for g, image in ranked[:TOP[dataset]]:
            box = best_crop(args.root, dataset, image)
            out, cps = preview(args.root, args.out, dataset, image, box)
            rows.append(dict(dataset=dataset, image=image, gain=g, box=box, preview=out, crop_psnr=cps,
                             psnr={m: per.get((dataset, image, m)) for m in TABLE}))
            print(f'{dataset} {image}: gain {g:+.2f} dB, crop {box}, {out}', flush=True)

    json.dump(rows, open(os.path.join(args.out, 'shortlist.json'), 'w'), indent=1)
    with open(os.path.join(args.out, 'shortlist.md'), 'w') as f:
        f.write('# Fig. 4 shortlist (x4)\n\n')
        f.write('Generated by `analysis/fig4_qualitative.py`. Gain = min(DeltaSR - HAT, DeltaSR - MambaIRv2), full-image '
                'Y-PSNR (dB). Crop = HR 160x160 window (40x40 LR, stride 16) with the largest local gain, HR Y std >= 10; '
                'box is x, y, w, h in HR pixels. "–" = model not finished yet.\n\n')
        f.write('| set | image | gain | crop gain | crop box (x, y, w, h) | ' + ' | '.join(TABLE) + ' | preview |\n')
        f.write('|---|---|---|---|---|' + '---|' * len(TABLE) + '---|\n')
        for r in rows:
            b = r['box']
            ps = ' | '.join('–' if r['psnr'][m] is None else f'{r["psnr"][m]:.2f}' for m in TABLE)
            f.write(f'| {r["dataset"]} | {r["image"]} | {r["gain"]:+.2f} | {b["crop_gain"]:+.2f} | '
                    f'{b["x"]}, {b["y"]}, {b["w"]}, {b["h"]} | {ps} | [png](preview/{r["dataset"]}_{r["image"]}.png) |\n')
    print('wrote', os.path.join(args.out, 'shortlist.md'))


if __name__ == '__main__':
    main()
