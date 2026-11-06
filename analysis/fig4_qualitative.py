"""Fig. 4 (x4 qualitative comparison): shortlist, best + alternative crop per image, previews, raw crops, contact sheet.

Reads docs/fig4_qualitative/per_image_x4.csv (analysis/per_image_metrics.py) and the SR images saved by each model's x4 test.
1. Rank images by gain = min(DeltaSR - HAT, DeltaSR - MambaIRv2) in full-image PSNR; keep the top 15 Urban100 + top 5 Manga109.
2. For each, slide a 160x160 HR window (= 40x40 LR, stride 16) over the image (inside crop_border 4). A window is eligible
   only if it has structure: HR Y std >= 10 and >= 25% of its pixels with Sobel gradient magnitude > 20 (HR Y, 0-255).
   Score = min(MSE_HAT - MSE_DeltaSR, MSE_MambaIRv2 - MSE_DeltaSR) on Y (0-255): the error DeltaSR removes, which favours
   visible errors (a PSNR gain is huge for tiny errors in flat regions). Best window = highest score; alternative = highest
   score among windows overlapping the best by <= 25% of the window area.
3. Write shortlist.md / shortlist.json, preview/<dataset>_<image>.png and ..._alt.png (HR with the box, 2x4 crops: HR +
   figure models, nearest-neighbour x2, model name and crop PSNR; grey "pending" tile for models without x4 outputs yet),
   crops/<dataset>_<image>/<model>.png and <model>_alt.png (raw 160x160, lossless), and contact_sheet.png (one row per
   image: HR, HAT, MambaIRv2, DeltaSR best crops).
Rankings and crops use DeltaSR, HAT and MambaIRv2 only, so re-running after DAT / CAT-A / MambaIR finish keeps the same images
and crops and only fills in their tiles.

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
WIN, STRIDE = 160, 16
MIN_STD, GRAD_THR, MIN_EDGE_FRAC, MAX_ALT_OVERLAP = 10.0, 20.0, 0.25, 0.25
TOP = {'Urban100': 15, 'Manga109': 5}
FIGURE = ['HR', 'SwinIR', 'HAT', 'DAT', 'CAT-A', 'MambaIR', 'MambaIRv2', 'DeltaSR']
TABLE = ['DeltaSR', 'MambaIRv2', 'HAT', 'DAT', 'CAT-A', 'MambaIR', 'SwinIR', 'RCAN', 'EDSR', 'SAN']
SHEET = ['HR', 'HAT', 'MambaIRv2', 'DeltaSR']
SUFFIX = '_x4ft250k'
TILE, LABEL_H = WIN * 2, 44
SHEET_MAX_W = 2400


def sr_path(root, model, dataset, image):
    p = os.path.join(root, 'results', MODELS[model], 'visualization', dataset,
                     LQ_TMPL[dataset].format(image) + SUFFIX + '.png')
    return p if os.path.isfile(p) else None


def read_hr(root, dataset, image, shape):
    hr = cv2.imread(os.path.join(root, 'datasets/SR', dataset, 'HR', image + '.png'), cv2.IMREAD_COLOR)
    return hr[:shape[0], :shape[1]]  # mod-crop to the SR size, as in PairedImageDataset


def y_of(img):
    # same Y conversion as calculate_psnr(test_y_channel=True): (H, W, 1) on 0-255
    return to_y_channel(img)[..., 0].astype(np.float64)


def window_means(a):
    """Means of a over all WIN x WIN windows at STRIDE (integral image)."""
    ii = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    h, w = a.shape
    ys = np.arange(0, h - WIN + 1, STRIDE)
    xs = np.arange(0, w - WIN + 1, STRIDE)
    Y, X = np.meshgrid(ys, xs, indexing='ij')
    s = ii[Y + WIN, X + WIN] - ii[Y, X + WIN] - ii[Y + WIN, X] + ii[Y, X]
    return s / WIN**2, ys, xs


def psnr_of(mse):
    return 10 * np.log10(255.0**2 / np.maximum(mse, 1e-10))


def pick_crops(root, dataset, image):
    imgs = {m: cv2.imread(sr_path(root, m, dataset, image), cv2.IMREAD_COLOR) for m in ('DeltaSR', 'HAT', 'MambaIRv2')}
    hr = read_hr(root, dataset, image, imgs['DeltaSR'].shape)
    b = BORDER
    hr_y = y_of(hr)[b:-b, b:-b]
    mse = {}
    for m, im in imgs.items():
        mse[m], ys, xs = window_means((y_of(im)[b:-b, b:-b] - hr_y) ** 2)
    m1, _, _ = window_means(hr_y)
    m2, _, _ = window_means(hr_y**2)
    std = np.sqrt(np.maximum(m2 - m1**2, 0))
    grad = np.hypot(cv2.Sobel(hr_y, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(hr_y, cv2.CV_64F, 0, 1, ksize=3))
    edge_frac, _, _ = window_means((grad > GRAD_THR).astype(np.float64))
    eligible = (std >= MIN_STD) & (edge_frac >= MIN_EDGE_FRAC)
    score = np.minimum(mse['HAT'] - mse['DeltaSR'], mse['MambaIRv2'] - mse['DeltaSR'])
    score = np.where(eligible, score, -np.inf)

    def box(r, c):
        return dict(x=int(xs[c] + b), y=int(ys[r] + b), w=WIN, h=WIN, score=float(score[r, c]),
                    gain_hat=float(psnr_of(mse['DeltaSR'][r, c]) - psnr_of(mse['HAT'][r, c])),
                    gain_v2=float(psnr_of(mse['DeltaSR'][r, c]) - psnr_of(mse['MambaIRv2'][r, c])),
                    hr_std=float(std[r, c]), edge_frac=float(edge_frac[r, c]))

    if not np.isfinite(score).any():
        return None, None, int(eligible.sum())
    r, c = np.unravel_index(np.argmax(score), score.shape)
    best = box(r, c)
    # alternative: best window overlapping the best one by at most MAX_ALT_OVERLAP of the area
    Y, X = np.meshgrid(ys, xs, indexing='ij')
    ov = np.maximum(0, WIN - np.abs(Y - ys[r])) * np.maximum(0, WIN - np.abs(X - xs[c])) / WIN**2
    alt_score = np.where(ov <= MAX_ALT_OVERLAP, score, -np.inf)
    alt = None
    if np.isfinite(alt_score).any():
        ra, ca = np.unravel_index(np.argmax(alt_score), alt_score.shape)
        alt = box(ra, ca)
    return best, alt, int(eligible.sum())


def font(size):
    for f in ('DejaVuSans.ttf', '/usr/share/fonts/dejavu/DejaVuSans.ttf', '/usr/share/fonts/TTF/DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def crops_of(root, dataset, image, bx):
    """{model: 160x160 BGR crop or None (pending)}, {model: crop Y-PSNR}"""
    x, y, w, h = bx['x'], bx['y'], bx['w'], bx['h']
    out, psnrs, hr = {}, {}, None
    for m in FIGURE[1:]:
        p = sr_path(root, m, dataset, image)
        if p is None:
            out[m] = None
            continue
        sr = cv2.imread(p, cv2.IMREAD_COLOR)
        if hr is None:
            hr = read_hr(root, dataset, image, sr.shape)
        out[m] = sr[y:y + h, x:x + w]
    out['HR'] = hr[y:y + h, x:x + w]
    hr_y = y_of(out['HR'])
    for m in FIGURE[1:]:
        if out[m] is not None:
            psnrs[m] = float(psnr_of(np.mean((y_of(out[m]) - hr_y) ** 2)))
    return out, psnrs, hr


def tile_of(img, label, f, red=False):
    tile = Image.new('RGB', (TILE, TILE + LABEL_H), 'white')
    if img is None:
        tile.paste(Image.new('RGB', (TILE, TILE), (170, 170, 170)), (0, 0))
        ImageDraw.Draw(tile).text((TILE // 2, TILE // 2), 'pending', fill='white', font=f, anchor='mm')
    else:
        tile.paste(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize((TILE, TILE), Image.NEAREST), (0, 0))
    ImageDraw.Draw(tile).text((TILE // 2, TILE + LABEL_H // 2), label, fill=(200, 0, 0) if red else 'black', font=f,
                              anchor='mm')
    return tile


def preview(root, out_dir, dataset, image, bx, tag):
    """tag '' (best) or '_alt'. Saves the raw crops and the preview panel; returns (path, crop PSNRs, crops)."""
    crops, psnrs, hr = crops_of(root, dataset, image, bx)
    crop_dir = os.path.join(out_dir, 'crops', f'{dataset}_{image}')
    os.makedirs(crop_dir, exist_ok=True)
    f = font(22)
    tiles = []
    for m in FIGURE:
        if crops[m] is not None:
            cv2.imwrite(os.path.join(crop_dir, f'{m}{tag}.png'), crops[m])
        label = 'HR' if m == 'HR' else (f'{m} (pending)' if crops[m] is None else f'{m}  {psnrs[m]:.2f} dB')
        tiles.append(tile_of(crops[m], label, f, red=m == 'DeltaSR'))
    right = Image.new('RGB', (4 * TILE, 2 * (TILE + LABEL_H)), 'white')
    for i, t in enumerate(tiles):
        right.paste(t, ((i % 4) * TILE, (i // 4) * (TILE + LABEL_H)))
    full = Image.fromarray(cv2.cvtColor(hr, cv2.COLOR_BGR2RGB))
    s = right.height / full.height
    full = full.resize((max(1, round(full.width * s)), right.height), Image.LANCZOS)
    x, y, w, h = bx['x'], bx['y'], bx['w'], bx['h']
    ImageDraw.Draw(full).rectangle([x * s, y * s, (x + w) * s, (y + h) * s], outline=(255, 0, 0), width=3)
    canvas = Image.new('RGB', (full.width + 16 + right.width, right.height), 'white')
    canvas.paste(full, (0, 0))
    canvas.paste(right, (full.width + 16, 0))
    os.makedirs(os.path.join(out_dir, 'preview'), exist_ok=True)
    out = os.path.join(out_dir, 'preview', f'{dataset}_{image}{tag}.png')
    canvas.save(out)
    return out, psnrs, crops


def contact_sheet(out_dir, rows):
    """One row per image: name + gains, then HR / HAT / MambaIRv2 / DeltaSR best crops (nearest x2)."""
    f, fs = font(22), font(18)
    name_w = 360
    row_h = TILE + LABEL_H
    sheet = Image.new('RGB', (name_w + len(SHEET) * TILE, (len(rows) + 1) * row_h // 1), 'white')
    d = ImageDraw.Draw(sheet)
    for j, m in enumerate(SHEET):
        d.text((name_w + j * TILE + TILE // 2, LABEL_H // 2), m, fill='black', font=f, anchor='mm')
    for i, r in enumerate(rows):
        y0 = LABEL_H + i * row_h
        b = r['box']
        d.text((10, y0 + 10), f'{r["dataset"]} {r["image"]}', fill='black', font=f)
        d.text((10, y0 + 44), f'image gain {r["gain"]:+.2f} dB', fill='black', font=fs)
        d.text((10, y0 + 72), f'crop vs HAT {b["gain_hat"]:+.2f} dB', fill='black', font=fs)
        d.text((10, y0 + 100), f'crop vs MambaIRv2 {b["gain_v2"]:+.2f} dB', fill='black', font=fs)
        d.text((10, y0 + 128), f'box {b["x"]},{b["y"]}', fill='black', font=fs)
        for j, m in enumerate(SHEET):
            img = r['crops'][m]
            label = 'HR' if m == 'HR' else f'{r["crop_psnr"][m]:.2f} dB'
            sheet.paste(tile_of(img, label, fs, red=m == 'DeltaSR'), (name_w + j * TILE, y0))
    sheet = sheet.crop((0, 0, sheet.width, LABEL_H + len(rows) * row_h))
    if sheet.width > SHEET_MAX_W:
        s = SHEET_MAX_W / sheet.width
        sheet = sheet.resize((SHEET_MAX_W, round(sheet.height * s)), Image.LANCZOS)
    out = os.path.join(out_dir, 'contact_sheet.png')
    sheet.save(out)
    return out


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
            ranked.append((min(d - per[(dataset, i, 'HAT')], d - per[(dataset, i, 'MambaIRv2')]), i))
        ranked.sort(reverse=True)
        for rank, (g, image) in enumerate(ranked[:TOP[dataset]], 1):
            best, alt, n_ok = pick_crops(args.root, dataset, image)
            if best is None:
                print(f'{dataset} {image}: no window with structure, skipped', flush=True)
                continue
            out, cps, crops = preview(args.root, args.out, dataset, image, best, '')
            alt_out = preview(args.root, args.out, dataset, image, alt, '_alt')[0] if alt else None
            rows.append(dict(dataset=dataset, image=image, rank=rank, gain=g, box=best, alt=alt, eligible=n_ok,
                             preview=out, preview_alt=alt_out, crop_psnr=cps, crops=crops,
                             psnr={m: per.get((dataset, image, m)) for m in TABLE}))
            print(f'{dataset} #{rank} {image}: gain {g:+.2f} dB, best {best}, alt {alt}', flush=True)

    sheet = contact_sheet(args.out, rows)
    json.dump([{k: v for k, v in r.items() if k != 'crops'} for r in rows],
              open(os.path.join(args.out, 'shortlist.json'), 'w'), indent=1)
    with open(os.path.join(args.out, 'shortlist.md'), 'w') as f:
        f.write('# Fig. 4 shortlist (x4)\n\n')
        f.write('Generated by `analysis/fig4_qualitative.py` (crop rule v2). Image gain = min(DeltaSR - HAT, DeltaSR - '
                'MambaIRv2), full-image Y-PSNR (dB); top 15 Urban100 + top 5 Manga109. Crop = HR 160x160 window (40x40 LR, '
                'stride 16) with HR Y std >= 10 and >= 25% of pixels with Sobel gradient > 20, maximising score = '
                'min(MSE_HAT - MSE_DeltaSR, MSE_MambaIRv2 - MSE_DeltaSR) (Y, 0-255). Alt = best window overlapping the best '
                'by <= 25%. Boxes are x, y, w, h in HR pixels. Contact sheet: `contact_sheet.png`.\n\n')
        f.write('| set | # | image | image gain | crop score (MSE) | crop vs HAT | crop vs MambaIRv2 | edge % | box | '
                'alt box (score, vs HAT / vs v2) | ' + ' | '.join(TABLE) + ' | previews |\n')
        f.write('|---|---|---|---|---|---|---|---|---|---|' + '---|' * len(TABLE) + '---|\n')
        for r in rows:
            b, a = r['box'], r['alt']
            alt_s = (f'{a["x"]}, {a["y"]} ({a["score"]:.1f}, {a["gain_hat"]:+.2f} / {a["gain_v2"]:+.2f})' if a else '–')
            ps = ' | '.join('–' if r['psnr'][m] is None else f'{r["psnr"][m]:.2f}' for m in TABLE)
            name = f'{r["dataset"]}_{r["image"]}'
            prev = f'[best](preview/{name}.png)' + (f', [alt](preview/{name}_alt.png)' if a else '')
            f.write(f'| {r["dataset"]} | {r["rank"]} | {r["image"]} | {r["gain"]:+.2f} | {b["score"]:.1f} | '
                    f'{b["gain_hat"]:+.2f} | {b["gain_v2"]:+.2f} | {100 * b["edge_frac"]:.0f} | {b["x"]}, {b["y"]}, '
                    f'{b["w"]}, {b["h"]} | {alt_s} | {ps} | {prev} |\n')
    print('wrote', os.path.join(args.out, 'shortlist.md'), 'and', sheet)


if __name__ == '__main__':
    main()
