"""Per-image PSNR / SSIM (x4, Urban100 + Manga109) of the same-setup models, from the SR images their test runs saved.

Same metric functions and settings as the tests (basicsr calculate_psnr / calculate_ssim, Y channel, crop_border = 4) on the
saved uint8 PNGs, which are the images the tests scored. Checks that each model's mean matches docs/results.csv (< 0.01 dB).
Writes docs/fig4_qualitative/per_image_x4.csv (dataset, image, model, psnr, ssim).

usage (on a compute node, from the repo root or a worktree):
  python analysis/per_image_metrics.py --root /speed-scratch/se_gham/GatedMambaIR [--workers 8]
"""
import argparse
import csv
import os
import sys
from multiprocessing import Pool

import cv2

sys.path.insert(0, '.')
from basicsr.metrics import calculate_psnr, calculate_ssim  # noqa: E402

SCALE = 4
SETS = ['Urban100', 'Manga109']
# model name -> test run name (results/<test name>/visualization/<set>/), results.csv test name
MODELS = {
    'DeltaSR': 'test_GatedMambaIR_SR_x4_lean_chgate_ft250k',
    'MambaIRv2': 'test_MambaIRv2_SR_x4_ft250k',
    'HAT': 'test_HAT_SR_x4_ft250k',
    'SwinIR': 'test_SwinIR_SR_x4_ft250k',
    'RCAN': 'test_RCAN_SR_x4_ft250k_gclip',
    'EDSR': 'test_EDSR_SR_x4_ft250k',
    'SAN': 'test_SAN_SR_x4_ft250k',
    'DAT': 'test_DAT_SR_x4_ft250k',
    'CAT-A': 'test_CAT_A_SR_x4_ft250k',
    'MambaIR': 'test_MambaIR_SR_x4_ft250k',
}
LQ_TMPL = {'Urban100': '{}x4', 'Manga109': '{}_LRBI_x4'}


def hr_name(dataset, sr_file):
    """img001x4_x4ft250k.png -> img001 ; Foo_LRBI_x4_x4ft250k.png -> Foo"""
    stem = os.path.splitext(sr_file)[0]
    lq = stem.rsplit('_', 1)[0]  # drop the test suffix
    tail = LQ_TMPL[dataset].format('')
    assert lq.endswith(tail), (dataset, sr_file)
    return lq[:-len(tail)]


def sr_files(root, test_name, dataset):
    d = os.path.join(root, 'results', test_name, 'visualization', dataset)
    if not os.path.isdir(d):
        return None
    return {hr_name(dataset, f): os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith('.png')}


def score(job):
    dataset, image, model, sr_path, hr_path = job
    sr = cv2.imread(sr_path, cv2.IMREAD_COLOR)
    hr = cv2.imread(hr_path, cv2.IMREAD_COLOR)
    hr = hr[:sr.shape[0], :sr.shape[1]]  # GT is mod-crop'ed to the SR size, as in PairedImageDataset
    assert sr.shape == hr.shape, (sr_path, sr.shape, hr.shape)
    kw = dict(crop_border=SCALE, input_order='HWC', test_y_channel=True)
    return dataset, image, model, calculate_psnr(sr, hr, **kw), calculate_ssim(sr, hr, **kw)


def table_means(root):
    """Newest results.csv row per (test name, dataset) -> (psnr, ssim)."""
    best = {}
    for r in csv.DictReader(open(os.path.join(root, 'docs', 'results.csv'))):
        k = (r['name'], r['dataset'])
        if r['psnr'] and (k not in best or r['timestamp'] > best[k][0]):
            best[k] = (r['timestamp'], float(r['psnr']), float(r['ssim'] or 'nan'))
    return {k: v[1:] for k, v in best.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/speed-scratch/se_gham/GatedMambaIR')
    p.add_argument('--out', default='docs/fig4_qualitative/per_image_x4.csv')
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()

    jobs = []
    for model, test_name in MODELS.items():
        for dataset in SETS:
            files = sr_files(args.root, test_name, dataset)
            if files is None:
                print(f'{model}: no saved {dataset} images yet ({test_name}), skipped')
                continue
            for image, path in files.items():
                jobs.append((dataset, image, model, path, os.path.join(args.root, 'datasets/SR', dataset, 'HR',
                                                                     image + '.png')))
    with Pool(args.workers) as pool:
        rows = sorted(pool.map(score, jobs, chunksize=4))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'image', 'model', 'psnr', 'ssim'])
        for d, i, m, ps, ss in rows:
            w.writerow([d, i, m, f'{ps:.4f}', f'{ss:.4f}'])
    print(f'wrote {len(rows)} rows to {args.out}')

    # mean check against the test numbers in results.csv
    ref = table_means(args.root)
    ok = True
    print(f'{"model":10s} {"set":9s} {"n":>4s} {"mean PSNR":>10s} {"table":>8s} {"diff":>7s} {"mean SSIM":>10s} {"table":>7s}')
    for model, test_name in MODELS.items():
        for dataset in SETS:
            vals = [(ps, ss) for d, i, m, ps, ss in rows if m == model and d == dataset]
            if not vals:
                continue
            mp = sum(v[0] for v in vals) / len(vals)
            ms = sum(v[1] for v in vals) / len(vals)
            tp, ts = ref.get((test_name, dataset), (float('nan'), float('nan')))
            flag = '' if abs(mp - tp) < 0.01 and abs(ms - ts) < 1e-4 else '  MISMATCH'
            ok &= not flag
            print(f'{model:10s} {dataset:9s} {len(vals):4d} {mp:10.4f} {tp:8.4f} {mp - tp:+7.4f} {ms:10.4f} {ts:7.4f}{flag}')
    print('mean check:', 'all match (< 0.01 dB)' if ok else 'MISMATCH, see above')


if __name__ == '__main__':
    main()
