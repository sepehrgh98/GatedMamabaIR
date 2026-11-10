"""Fig. 4 search v3 (x4): windows where DeltaSR is visibly better than ALL six figure baselines, over every image.

No inference: reads the SR images saved by the x4 tests (same runs as docs/fig4_qualitative/per_image_x4.csv).
For every Urban100 and Manga109 image:
1. Slide an 80x80 HR window, stride 8, on Y (0-255, same conversion as calculate_psnr(test_y_channel=True)). Window corners
   are absolute HR pixel coordinates on multiples of 8, with the window inside the crop_border = 4 of the tests.
2. Eligible if HR std >= 10, >= 25% of pixels with Sobel gradient magnitude > 20 (Sobel on the full HR Y), and DeltaSR
   window PSNR >= 24 dB.
3. gain = DeltaSR window PSNR - best of the six baselines (SwinIR, HAT, DAT, CAT-A, MambaIR, MambaIRv2); keep gain >= 1.5 dB.
4. score = min over the six baselines of (MSE_baseline - MSE_DeltaSR); keep at most 2 windows per image, greedy by score,
   overlapping each other by <= 25% of the window area.
Outputs in docs/fig4_qualitative/v3/: candidates.csv (all kept windows, sorted by gain), contact_sheet*.png (row 0 = the
reference window Urban100 img044 (720, 240), then the top 30 by gain; HR + 7 models, nearest x3, window PSNR under each tile),
crops/<dataset>_<image>_<x>_<y>/<model>.png (raw 80x80, lossless), full_hr/<dataset>_<image>.png (HR mod-cropped to the SR
size, i.e. in the same coordinates as the boxes, no box).

usage (on a compute node, from the repo root or a worktree):
  python analysis/fig4_search_v3.py --root /speed-scratch/se_gham/GatedMambaIR [--workers 8]
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
from fig4_qualitative import font, psnr_of, read_hr, sr_path, y_of  # noqa: E402
from per_image_metrics import SETS  # noqa: E402

BORDER = 4
WIN, STRIDE = 80, 8
MIN_STD, GRAD_THR, MIN_EDGE_FRAC = 10.0, 20.0, 0.25
MIN_DELTA_PSNR, MIN_GAIN, MAX_OVERLAP, PER_IMAGE = 24.0, 1.5, 0.25, 2
TOP_N = 30
BASELINES = ['SwinIR', 'HAT', 'DAT', 'CAT-A', 'MambaIR', 'MambaIRv2']
MODELS7 = BASELINES + ['DeltaSR']
COLS = ['HR'] + MODELS7
REF = dict(dataset='Urban100', image='img044', x=720, y=240)
UP, LABEL_H, NAME_W = 3, 34, 300
TILE = WIN * UP
SHEET_MAX_W, SHEET_MAX_H = 2400, 6000


def window_sums(ii, ys, xs):
    Y, X = np.meshgrid(ys, xs, indexing='ij')
    return ii[Y + WIN, X + WIN] - ii[Y, X + WIN] - ii[Y + WIN, X] + ii[Y, X]


def integral(a):
    return np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def load(root, dataset, image):
    srs = {m: cv2.imread(sr_path(root, m, dataset, image), cv2.IMREAD_COLOR) for m in MODELS7}
    shape = srs['DeltaSR'].shape
    for m, im in srs.items():
        assert im is not None and im.shape == shape, (dataset, image, m)
    return read_hr(root, dataset, image, shape), srs


def search(job):
    root, dataset, image = job
    hr, srs = load(root, dataset, image)
    h, w = hr.shape[:2]
    first = -(-BORDER // STRIDE) * STRIDE  # first multiple of STRIDE >= BORDER
    ys = np.arange(first, h - BORDER - WIN + 1, STRIDE)
    xs = np.arange(first, w - BORDER - WIN + 1, STRIDE)
    if len(ys) == 0 or len(xs) == 0:
        return []
    hr_y = y_of(hr)
    n = WIN * WIN
    mse = {m: window_sums(integral((y_of(im) - hr_y) ** 2), ys, xs) / n for m, im in srs.items()}
    m1 = window_sums(integral(hr_y), ys, xs) / n
    m2 = window_sums(integral(hr_y ** 2), ys, xs) / n
    std = np.sqrt(np.maximum(m2 - m1 ** 2, 0))
    grad = np.hypot(cv2.Sobel(hr_y, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(hr_y, cv2.CV_64F, 0, 1, ksize=3))
    edge = window_sums(integral((grad > GRAD_THR).astype(np.float64)), ys, xs) / n
    psnr = {m: psnr_of(v) for m, v in mse.items()}
    best_base = np.max(np.stack([psnr[m] for m in BASELINES]), 0)
    gain = psnr['DeltaSR'] - best_base
    score = np.min(np.stack([mse[m] - mse['DeltaSR'] for m in BASELINES]), 0)
    ok = (std >= MIN_STD) & (edge >= MIN_EDGE_FRAC) & (psnr['DeltaSR'] >= MIN_DELTA_PSNR) & (gain >= MIN_GAIN)
    score = np.where(ok, score, -np.inf)
    Y, X = np.meshgrid(ys, xs, indexing='ij')
    out = []
    while len(out) < PER_IMAGE and np.isfinite(score).any():
        r, c = np.unravel_index(np.argmax(score), score.shape)
        win_psnr = {m: float(psnr[m][r, c]) for m in MODELS7}
        out.append(dict(dataset=dataset, image=image, x=int(xs[c]), y=int(ys[r]), w=WIN, h=WIN, gain=float(gain[r, c]),
                        score=float(score[r, c]), hr_std=float(std[r, c]), edge_frac=float(edge[r, c]),
                        best_win_baseline=max(BASELINES, key=lambda m: win_psnr[m]), psnr=win_psnr))
        ov = np.maximum(0, WIN - np.abs(Y - ys[r])) * np.maximum(0, WIN - np.abs(X - xs[c])) / n
        score = np.where(ov > MAX_OVERLAP, -np.inf, score)
    return out


def window_crops(root, c):
    hr, srs = load(root, c['dataset'], c['image'])
    x, y = c['x'], c['y']
    crops = {m: im[y:y + WIN, x:x + WIN] for m, im in srs.items()}
    crops['HR'] = hr[y:y + WIN, x:x + WIN]
    hr_y = y_of(crops['HR'])
    psnr = {m: float(psnr_of(np.mean((y_of(crops[m]) - hr_y) ** 2))) for m in MODELS7}
    return crops, psnr, hr


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
    paths = []
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
            lines = [(f'#{r["rank"]}' if r['rank'] else 'ref (row 0)', f),
                     (f'{r["dataset"]} {r["image"]}', f),
                     (f'box x={r["x"]} y={r["y"]} {WIN}x{WIN}', fs),
                     (f'gain {r["gain"]:+.2f} dB', fs),
                     (f'vs best {r["best_win_baseline"]}', fs),
                     (f'image: DeltaSR {r["full_delta"]:.2f}', fs),
                     (f'  best base {r["full_best"]:.2f} ({r["full_best_model"]})', fs)]
            for li, (t, ff) in enumerate(lines):
                d.text((10, y0 + 12 + li * 30), t, fill=(200, 0, 0) if li == 6 and r['full_best'] > r['full_delta']
                       else 'black', font=ff)
            for j, m in enumerate(COLS):
                label = 'HR' if m == 'HR' else f'{r["psnr"][m]:.2f} dB'
                sheet.paste(tile_of(r['crops'][m], label, fs, red=m == 'DeltaSR'), (NAME_W + j * TILE, y0))
        name = 'contact_sheet.png' if k == 0 else f'contact_sheet_{k // per_sheet + 1}.png'
        sheet.save(os.path.join(out_dir, name))
        paths.append(name)
    return paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/speed-scratch/se_gham/GatedMambaIR')
    p.add_argument('--per_image', default='docs/fig4_qualitative/per_image_x4.csv')
    p.add_argument('--out', default='docs/fig4_qualitative/v3')
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()

    full = {}
    for r in csv.DictReader(open(args.per_image)):
        full[(r['dataset'], r['image'], r['model'])] = float(r['psnr'])
    jobs = [(args.root, d, i) for d in SETS for i in sorted({i for dd, i, m in full if dd == d and m == 'DeltaSR'})]
    print(f'{len(jobs)} images', flush=True)
    with Pool(args.workers) as pool:
        cands = [c for res in pool.imap(search, jobs, chunksize=2) for c in res]
    cands.sort(key=lambda c: -c['gain'])
    for c in cands:
        c['full_delta'] = full[(c['dataset'], c['image'], 'DeltaSR')]
        c['full_best_model'] = max(BASELINES, key=lambda m: full[(c['dataset'], c['image'], m)])
        c['full_best'] = full[(c['dataset'], c['image'], c['full_best_model'])]

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'candidates.csv'), 'w', newline='') as fh:
        wr = csv.writer(fh)
        wr.writerow(['rank', 'dataset', 'image', 'x', 'y', 'w', 'h', 'gain', 'score', 'best_win_baseline', 'hr_std',
                     'edge_frac'] + [f'psnr_{m}' for m in MODELS7] +
                    ['full_psnr_DeltaSR', 'full_psnr_best_baseline', 'full_best_baseline', 'full_delta_minus_best'])
        for k, c in enumerate(cands, 1):
            wr.writerow([k, c['dataset'], c['image'], c['x'], c['y'], c['w'], c['h'], f'{c["gain"]:.3f}',
                         f'{c["score"]:.2f}', c['best_win_baseline'], f'{c["hr_std"]:.1f}', f'{c["edge_frac"]:.3f}'] +
                        [f'{c["psnr"][m]:.3f}' for m in MODELS7] +
                        [f'{c["full_delta"]:.4f}', f'{c["full_best"]:.4f}', c['full_best_model'],
                         f'{c["full_delta"] - c["full_best"]:+.4f}'])
    print(f'{len(cands)} candidate windows in {len({(c["dataset"], c["image"]) for c in cands})} images', flush=True)

    # reference window (always row 0), then the top 30 by gain
    ref = dict(REF, w=WIN, h=WIN, rank=0)
    rows = [ref]
    for k, c in enumerate(cands[:TOP_N], 1):
        rows.append(dict(c, rank=k))
    os.makedirs(os.path.join(args.out, 'full_hr'), exist_ok=True)
    for r in rows:
        crops, psnr, hr = window_crops(args.root, r)
        r['crops'] = crops
        if r['rank'] == 0:  # the reference is not necessarily in the candidate list: compute its numbers here
            r['psnr'] = psnr
            r['best_win_baseline'] = max(BASELINES, key=lambda m: psnr[m])
            r['gain'] = psnr['DeltaSR'] - psnr[r['best_win_baseline']]
            r['full_delta'] = full[(r['dataset'], r['image'], 'DeltaSR')]
            r['full_best_model'] = max(BASELINES, key=lambda m: full[(r['dataset'], r['image'], m)])
            r['full_best'] = full[(r['dataset'], r['image'], r['full_best_model'])]
        else:  # crop PSNR must equal the windowed search value
            assert all(abs(psnr[m] - r['psnr'][m]) < 1e-6 for m in MODELS7), (r['image'], psnr, r['psnr'])
        d = os.path.join(args.out, 'crops', f'{r["dataset"]}_{r["image"]}_{r["x"]}_{r["y"]}')
        os.makedirs(d, exist_ok=True)
        for m in COLS:
            cv2.imwrite(os.path.join(d, f'{m}.png'), crops[m])
        fp = os.path.join(args.out, 'full_hr', f'{r["dataset"]}_{r["image"]}.png')
        if not os.path.exists(fp):
            cv2.imwrite(fp, hr)
        print(f'{"ref" if r["rank"] == 0 else "#%d" % r["rank"]:>4} {r["dataset"]:8s} {r["image"]:28s} '
              f'({r["x"]},{r["y"]}) gain {r["gain"]:+.2f} vs {r["best_win_baseline"]:9s} DeltaSR {r["psnr"]["DeltaSR"]:.2f} '
              f'| image DeltaSR {r["full_delta"]:.2f} best {r["full_best_model"]} {r["full_best"]:.2f}'
              f'{"  <-- baseline wins image" if r["full_best"] > r["full_delta"] else ""}', flush=True)
    print('sheets:', sheets(args.out, rows))


if __name__ == '__main__':
    main()
