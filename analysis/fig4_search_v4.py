"""Fig. 4 search v4 (x4): windows where every baseline loses a fine pattern and DeltaSR keeps it. No inference.

Reads the SR images saved by the x4 tests (same runs as v3 / docs/fig4_qualitative/per_image_x4.csv).
For every Urban100 and Manga109 image, HR windows of 80, 64 and 48 px (stride 8, corners on absolute multiples of 8, inside
crop_border = 4), Y channel 0-255 (basicsr to_y_channel):
1. hp(X) = X - GaussianBlur(X, sigma=1) (cv2, kernel 7x7, computed on the full image).
   NHE(model) = sum((hp(model) - hp(HR))^2) / sum(hp(HR)^2) over the window.
2. Eligible: std(hp(HR)) >= T1; DeltaSR PSNR >= 22; DeltaSR PSNR >= median(6 baseline PSNRs) + T2; gain = DeltaSR PSNR -
   best baseline PSNR >= T3 (> 0, so DeltaSR beats all six).
3. score = min_b NHE(b) - NHE(DeltaSR) >= T4; ratio = NHE(DeltaSR) / min_b NHE(b).
   Spec values T1..T4 = 10, 5, 2, 0.08 (defaults) admit no window; fig4_search_v4.sh runs 5, 2.5, 1.5, 0.05 (see TH).
4. Per image, greedy by score over all three sizes, at most 2 windows, overlap (intersection / smaller window area) <= 25%:
   this keeps the best size per location. img044, img063, img067 (Urban100) are excluded; their v3 reference windows are
   scored separately (on the full image, as the search does, and on the saved 80x80 crop alone).
Window PSNR / SSIM (Y) of each model are computed for the kept windows (SSIM: basicsr calculate_ssim on the crop).
Outputs in docs/fig4_qualitative/v4/: candidates.csv, references.csv, contact_sheet*.png (3 references, then top 25 Urban100
and top 15 Manga109 by score; tiles 240 px nearest, i.e. x3 for 80 px windows), crops/<dataset>_<image>_<x>_<y>/<model>.png
(raw window at its own size, lossless), full_hr/<dataset>_<image>.png (HR mod-cropped to the SR size, no box).

usage (on a compute node, from the repo root or a worktree):
  python analysis/fig4_search_v4.py --root /speed-scratch/se_gham/GatedMambaIR [--workers 8]
"""
import argparse
import csv
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from basicsr.metrics import calculate_ssim  # noqa: E402
from fig4_qualitative import font, psnr_of, y_of  # noqa: E402
from fig4_search_v3 import BASELINES, MODELS7, integral, load  # noqa: E402
from per_image_metrics import SETS  # noqa: E402

BORDER, STRIDE, SIZES = 4, 8, (80, 64, 48)
SIGMA = 1.0
# defaults = the v4 spec; it admits no window at all (nor the three references), so the launcher passes the loosest
# thresholds that admit all three references: hp_std 5 (img044 7.2), over_median 2.5 (img067 2.60), gain 1.5 (img067
# 1.78), score 0.05 (img067 0.056). Workers get them through this module-level dict (fork).
TH = dict(hp_std=10.0, delta_psnr=22.0, over_median=5.0, gain=2.0, score=0.08)
MAX_OVERLAP, PER_IMAGE = 0.25, 2
TOP = {'Urban100': 25, 'Manga109': 15}
EXCLUDE = {('Urban100', 'img044'), ('Urban100', 'img063'), ('Urban100', 'img067')}
REFS = [('Urban100', 'img044', 720, 240), ('Urban100', 'img063', 88, 40), ('Urban100', 'img067', 616, 128)]
COLS = ['HR'] + MODELS7
TILE, LABEL_H, NAME_W = 240, 34, 300
SHEET_MAX_W, SHEET_MAX_H = 2400, 6000


def hp(y):
    return y - cv2.GaussianBlur(y, (0, 0), SIGMA)


def wsum(ii, ys, xs, w):
    Y, X = np.meshgrid(ys, xs, indexing='ij')
    return ii[Y + w, X + w] - ii[Y, X + w] - ii[Y + w, X] + ii[Y, X]


def window_stats(y, w):
    """y: {'HR': Y, model: Y}. Returns dict of per-window arrays for window size w (all positions)."""
    h, W = y['HR'].shape
    first = -(-BORDER // STRIDE) * STRIDE
    ys = np.arange(first, h - BORDER - w + 1, STRIDE)
    xs = np.arange(first, W - BORDER - w + 1, STRIDE)
    return ys, xs


def score_window(y, hpy, x0, y0, w):
    """NHE / PSNR of each model on one window (for references; same definitions as the search)."""
    sl = np.s_[y0:y0 + w, x0:x0 + w]
    den = np.sum(hpy['HR'][sl] ** 2)
    nhe = {m: float(np.sum((hpy[m][sl] - hpy['HR'][sl]) ** 2) / den) for m in MODELS7}
    psnr = {m: float(psnr_of(np.mean((y[m][sl] - y['HR'][sl]) ** 2))) for m in MODELS7}
    return nhe, psnr, float(np.std(hpy['HR'][sl]))


def search(job):
    root, dataset, image = job
    hr, srs = load(root, dataset, image)
    y = {m: y_of(im) for m, im in srs.items()}
    y['HR'] = y_of(hr)
    hpy = {k: hp(v) for k, v in y.items()}
    ii_hp2 = integral(hpy['HR'] ** 2)
    ii_hp = integral(hpy['HR'])
    ii_e = {m: integral((hpy[m] - hpy['HR']) ** 2) for m in MODELS7}
    ii_mse = {m: integral((y[m] - y['HR']) ** 2) for m in MODELS7}
    cands = []
    for w in SIZES:
        ys, xs = window_stats(y, w)
        if len(ys) == 0 or len(xs) == 0:
            continue
        n = w * w
        den = wsum(ii_hp2, ys, xs, w)
        hp_std = np.sqrt(np.maximum(den / n - (wsum(ii_hp, ys, xs, w) / n) ** 2, 0))
        nhe = {m: wsum(ii_e[m], ys, xs, w) / np.maximum(den, 1e-10) for m in MODELS7}
        psnr = {m: psnr_of(wsum(ii_mse[m], ys, xs, w) / n) for m in MODELS7}
        base = np.stack([psnr[m] for m in BASELINES])
        gain = psnr['DeltaSR'] - base.max(0)
        over_med = psnr['DeltaSR'] - np.median(base, 0)
        base_nhe = np.stack([nhe[m] for m in BASELINES])
        score = base_nhe.min(0) - nhe['DeltaSR']
        t = TH
        ok = ((hp_std >= t['hp_std']) & (psnr['DeltaSR'] >= t['delta_psnr']) & (over_med >= t['over_median']) &
              (gain >= t['gain']) & (score >= t['score']))
        for r, c in zip(*np.nonzero(ok)):
            cands.append(dict(dataset=dataset, image=image, x=int(xs[c]), y=int(ys[r]), w=w, h=w,
                              score=float(score[r, c]), ratio=float(nhe['DeltaSR'][r, c] / base_nhe[:, r, c].min()),
                              gain=float(gain[r, c]), over_median=float(over_med[r, c]), hp_std=float(hp_std[r, c]),
                              nhe={m: float(nhe[m][r, c]) for m in MODELS7},
                              psnr={m: float(psnr[m][r, c]) for m in MODELS7}))
    cands.sort(key=lambda c: -c['score'])
    kept = []
    for c in cands:
        if len(kept) == PER_IMAGE:
            break
        if all(overlap(c, k) <= MAX_OVERLAP for k in kept):
            kept.append(c)
    for c in kept:
        c['best_win_baseline'] = max(BASELINES, key=lambda m: c['psnr'][m])
        c['ssim'] = ssim_of(hr, srs, c)
    return kept


def overlap(a, b):
    ix = max(0, min(a['x'] + a['w'], b['x'] + b['w']) - max(a['x'], b['x']))
    iy = max(0, min(a['y'] + a['h'], b['y'] + b['h']) - max(a['y'], b['y']))
    return ix * iy / min(a['w'] * a['h'], b['w'] * b['h'])


def ssim_of(hr, srs, c):
    sl = np.s_[c['y']:c['y'] + c['h'], c['x']:c['x'] + c['w']]
    return {m: float(calculate_ssim(srs[m][sl], hr[sl], crop_border=0, test_y_channel=True)) for m in MODELS7}


def reference(root, dataset, image, x, y0, w=80):
    hr, srs = load(root, dataset, image)
    yy = {m: y_of(im) for m, im in srs.items()}
    yy['HR'] = y_of(hr)
    # (a) as in the search: high-pass on the full image
    nhe, psnr, hp_std = score_window(yy, {k: hp(v) for k, v in yy.items()}, x, y0, w)
    # (b) high-pass on the 80x80 crop alone (as on the saved v3 crops)
    sl = np.s_[y0:y0 + w, x:x + w]
    cy = {k: v[sl] for k, v in yy.items()}
    nhe_c, _, _ = score_window(cy, {k: hp(v) for k, v in cy.items()}, 0, 0, w)
    base = [psnr[m] for m in BASELINES]
    c = dict(dataset=dataset, image=image, x=x, y=y0, w=w, h=w, nhe=nhe, psnr=psnr, hp_std=hp_std,
             score=min(nhe[m] for m in BASELINES) - nhe['DeltaSR'],
             ratio=nhe['DeltaSR'] / min(nhe[m] for m in BASELINES),
             score_crop=min(nhe_c[m] for m in BASELINES) - nhe_c['DeltaSR'],
             gain=psnr['DeltaSR'] - max(base), over_median=psnr['DeltaSR'] - float(np.median(base)),
             best_win_baseline=max(BASELINES, key=lambda m: psnr[m]))
    c['ssim'] = ssim_of(hr, srs, c)
    return c


def tile_of(img, label, f, red=False):
    tile = Image.new('RGB', (TILE, TILE + LABEL_H), 'white')
    tile.paste(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize((TILE, TILE), Image.NEAREST), (0, 0))
    ImageDraw.Draw(tile).text((TILE // 2, TILE + LABEL_H // 2), label, fill=(200, 0, 0) if red else 'black', font=f,
                              anchor='mm')
    return tile


def sheets(out_dir, rows):
    f, fs = font(22), font(19)
    row_h = TILE + LABEL_H
    width = NAME_W + len(COLS) * TILE
    assert width <= SHEET_MAX_W, width
    per_sheet = (SHEET_MAX_H - LABEL_H) // row_h
    names = []
    for k in range(0, len(rows), per_sheet):
        chunk = rows[k:k + per_sheet]
        sheet = Image.new('RGB', (width, LABEL_H + len(chunk) * row_h), 'white')
        d = ImageDraw.Draw(sheet)
        for j, m in enumerate(COLS):
            d.text((NAME_W + j * TILE + TILE // 2, LABEL_H // 2), m, fill=(200, 0, 0) if m == 'DeltaSR' else 'black',
                   font=f, anchor='mm')
        for i, r in enumerate(chunk):
            y0 = LABEL_H + i * row_h
            d.line([(0, y0), (width, y0)], fill=(200, 200, 200))
            flag = r['full_best'] > r['full_delta']
            lines = [(r['tag'], f), (f'{r["dataset"]} {r["image"]}', f),
                     (f'box x={r["x"]} y={r["y"]} {r["w"]}x{r["h"]}', fs),
                     (f'score {r["score"]:.3f}  ratio {r["ratio"]:.2f}', fs),
                     (f'gain {r["gain"]:+.2f} vs {r["best_win_baseline"]}', fs),
                     (f'image: DeltaSR {r["full_delta"]:.2f}', fs),
                     (f'  best base {r["full_best"]:.2f} ({r["full_best_model"]})', fs)]
            for li, (t, ff) in enumerate(lines):
                d.text((10, y0 + 12 + li * 30), t, fill=(200, 0, 0) if li == 6 and flag else 'black', font=ff)
            for j, m in enumerate(COLS):
                label = 'HR' if m == 'HR' else f'{r["psnr"][m]:.2f} dB'
                sheet.paste(tile_of(r['crops'][m], label, fs, red=m == 'DeltaSR'), (NAME_W + j * TILE, y0))
        name = 'contact_sheet.png' if k == 0 else f'contact_sheet_{k // per_sheet + 1}.png'
        sheet.save(os.path.join(out_dir, name))
        names.append(name)
    return names


def add_full(c, full):
    c['full_delta'] = full[(c['dataset'], c['image'], 'DeltaSR')]
    c['full_best_model'] = max(BASELINES, key=lambda m: full[(c['dataset'], c['image'], m)])
    c['full_best'] = full[(c['dataset'], c['image'], c['full_best_model'])]


def write_csv(path, rows, extra=()):
    with open(path, 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['rank', 'dataset', 'image', 'x', 'y', 'w', 'h', 'score', 'ratio'] + list(extra) +
                    ['gain', 'best_win_baseline', 'over_median', 'hp_std'] + [f'nhe_{m}' for m in MODELS7] +
                    [f'psnr_{m}' for m in MODELS7] + [f'ssim_{m}' for m in MODELS7] +
                    ['full_psnr_DeltaSR', 'full_psnr_best_baseline', 'full_best_baseline', 'full_delta_minus_best'])
        for c in rows:
            wr.writerow([c.get('rank', ''), c['dataset'], c['image'], c['x'], c['y'], c['w'], c['h'], f'{c["score"]:.4f}',
                         f'{c["ratio"]:.3f}'] + [f'{c[e]:.4f}' for e in extra] +
                        [f'{c["gain"]:.3f}', c['best_win_baseline'], f'{c["over_median"]:.3f}', f'{c["hp_std"]:.2f}'] +
                        [f'{c["nhe"][m]:.4f}' for m in MODELS7] + [f'{c["psnr"][m]:.3f}' for m in MODELS7] +
                        [f'{c["ssim"][m]:.4f}' for m in MODELS7] +
                        [f'{c["full_delta"]:.4f}', f'{c["full_best"]:.4f}', c['full_best_model'],
                         f'{c["full_delta"] - c["full_best"]:+.4f}'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/speed-scratch/se_gham/GatedMambaIR')
    p.add_argument('--per_image', default='docs/fig4_qualitative/per_image_x4.csv')
    p.add_argument('--out', default='docs/fig4_qualitative/v4')
    p.add_argument('--workers', type=int, default=8)
    for k, v in TH.items():
        p.add_argument('--min_' + k, type=float, default=v)
    args = p.parse_args()
    TH.update({k: getattr(args, 'min_' + k) for k in TH})
    print('thresholds:', TH, flush=True)

    full = {}
    for r in csv.DictReader(open(args.per_image)):
        full[(r['dataset'], r['image'], r['model'])] = float(r['psnr'])
    os.makedirs(args.out, exist_ok=True)

    refs = []
    for ds, im, x, y in REFS:
        c = reference(args.root, ds, im, x, y)
        add_full(c, full)
        refs.append(c)
        print(f'REF {ds} {im} ({x},{y}): score {c["score"]:.3f} (hp on crop {c["score_crop"]:.3f}) ratio {c["ratio"]:.2f} '
              f'gain {c["gain"]:+.2f} over median {c["over_median"]:+.2f} hp_std {c["hp_std"]:.1f}', flush=True)
    write_csv(os.path.join(args.out, 'references.csv'), refs, extra=('score_crop',))

    jobs = [(args.root, d, i) for d in SETS for i in sorted({i for dd, i, m in full if dd == d and m == 'DeltaSR'})
            if (d, i) not in EXCLUDE]
    print(f'{len(jobs)} images', flush=True)
    with Pool(args.workers) as pool:
        cands = [c for res in pool.imap(search, jobs, chunksize=2) for c in res]
    cands.sort(key=lambda c: -c['score'])
    rank = {d: 0 for d in SETS}
    for c in cands:
        add_full(c, full)
        rank[c['dataset']] += 1
        c['rank'] = rank[c['dataset']]
    write_csv(os.path.join(args.out, 'candidates.csv'), cands)
    print(f'{len(cands)} windows in {len({(c["dataset"], c["image"]) for c in cands})} images '
          f'({sum(c["dataset"] == "Urban100" for c in cands)} Urban100, {sum(c["dataset"] == "Manga109" for c in cands)} '
          'Manga109)', flush=True)

    rows = [dict(c, tag=f'ref {k + 1}') for k, c in enumerate(refs)]
    for d in SETS:
        rows += [dict(c, tag=f'{d} #{c["rank"]}') for c in cands if c['dataset'] == d][:TOP[d]]
    os.makedirs(os.path.join(args.out, 'full_hr'), exist_ok=True)
    for r in rows:
        hr, srs = load(args.root, r['dataset'], r['image'])
        sl = np.s_[r['y']:r['y'] + r['h'], r['x']:r['x'] + r['w']]
        r['crops'] = {m: im[sl] for m, im in srs.items()}
        r['crops']['HR'] = hr[sl]
        d = os.path.join(args.out, 'crops', f'{r["dataset"]}_{r["image"]}_{r["x"]}_{r["y"]}')
        os.makedirs(d, exist_ok=True)
        for m in COLS:
            cv2.imwrite(os.path.join(d, f'{m}.png'), r['crops'][m])
        fp = os.path.join(args.out, 'full_hr', f'{r["dataset"]}_{r["image"]}.png')
        if not os.path.exists(fp):
            cv2.imwrite(fp, hr)
        print(f'{r["tag"]:>13} {r["image"]:28s} ({r["x"]},{r["y"]},{r["w"]}) score {r["score"]:.3f} ratio {r["ratio"]:.2f} '
              f'gain {r["gain"]:+.2f} vs {r["best_win_baseline"]:9s} DeltaSR {r["psnr"]["DeltaSR"]:.2f} | image DeltaSR '
              f'{r["full_delta"]:.2f} best {r["full_best_model"]} {r["full_best"]:.2f}'
              f'{"  <-- baseline wins image" if r["full_best"] > r["full_delta"] else ""}', flush=True)
    print('sheets:', sheets(args.out, rows))


if __name__ == '__main__':
    main()
