"""Regenerate bicubic LR images from HR with BasicSR's MATLAB-imresize port (the SR benchmark standard).

    python basicsr/tools/generate_bicubic_lr.py --hr datasets/DF2K/DF2K/HR \
        --out datasets/DF2K/DF2K/LR_bicubic --scales 2 3 4 --workers 32

Writes <out>/X<s>/<name>x<s>.png, matching filename_tmpl '{}x<s>' in the configs. Existing outputs are skipped.
"""
import argparse
import glob
import importlib.util
import os
from multiprocessing import Pool
from os import path as osp

import cv2
import numpy as np

# load matlab_functions by path: importing the basicsr package pulls in torch archs and mamba_ssm
_spec = importlib.util.spec_from_file_location(
    'matlab_functions', osp.join(osp.dirname(__file__), osp.pardir, 'utils', 'matlab_functions.py'))
_mf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mf)


def process(job):
    hr_path, out_dir, scales = job
    name = osp.splitext(osp.basename(hr_path))[0]
    todo = [s for s in scales if not osp.exists(osp.join(out_dir, f'X{s}', f'{name}x{s}.png'))]
    if not todo:
        return 0
    img = cv2.imread(hr_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
    h, w = img.shape[:2]
    for s in todo:
        assert h % s == 0 and w % s == 0, f'{hr_path}: {w}x{h} not divisible by {s}'
        lr = _mf.imresize(img, 1 / s, antialiasing=True)
        lr = np.clip(np.round(lr * 255.), 0, 255).astype(np.uint8)
        assert lr.shape[:2] == (h // s, w // s), (hr_path, lr.shape)
        cv2.imwrite(osp.join(out_dir, f'X{s}', f'{name}x{s}.png'), lr)
    return len(todo)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hr', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--scales', type=int, nargs='+', default=[2, 3, 4])
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    args = parser.parse_args()

    for s in args.scales:
        os.makedirs(osp.join(args.out, f'X{s}'), exist_ok=True)
    hr_paths = sorted(glob.glob(osp.join(args.hr, '*.png')))
    jobs = [(p, args.out, args.scales) for p in hr_paths]
    written = 0
    with Pool(args.workers) as pool:
        for i, n in enumerate(pool.imap_unordered(process, jobs, chunksize=4), 1):
            written += n
            if i % 200 == 0 or i == len(jobs):
                print(f'{i}/{len(jobs)} HR images done, {written} LR images written', flush=True)


if __name__ == '__main__':
    main()
