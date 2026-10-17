"""Append test results to a shared CSV so experiments can be compared later.

One row per (test run, dataset). Read it with ``basicsr/tools/compare_results.py``.
"""
import csv
import os
import re
import subprocess
from datetime import datetime
from os import path as osp

FIELDS = [
    'timestamp', 'name', 'model', 'scale', 'dataset', 'psnr', 'ssim', 'n_images', 'runtime_s', 'iter', 'checkpoint',
    'config', 'git_commit', 'git_dirty', 'tag'
]


def git_info(root_path):
    """Return (commit, dirty) for the repo, or ('unknown', '') if git fails."""
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=root_path,
                                         stderr=subprocess.DEVNULL).decode().strip()
        status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root_path,
                                         stderr=subprocess.DEVNULL).decode().strip()
        return commit, 'yes' if status else 'no'
    except (subprocess.CalledProcessError, OSError):
        return 'unknown', ''


def checkpoint_iter(checkpoint):
    """Parse the iteration from names like net_g_464000.pth; '' if absent."""
    match = re.search(r'_(\d+)\.pth$', checkpoint or '')
    return match.group(1) if match else ''


def append_rows(csv_path, rows):
    """Append rows (dicts keyed by FIELDS) to csv_path, writing the header if the file is new."""
    os.makedirs(osp.dirname(osp.abspath(csv_path)), exist_ok=True)
    is_new = not osp.exists(csv_path) or osp.getsize(csv_path) == 0
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in FIELDS})


def make_row(opt, dataset_name, metrics, n_images, runtime_s, config_path, git, tag=''):
    """Build one CSV row from a parsed test opt and the averaged metric values."""
    checkpoint = opt['path'].get('pretrain_network_g') or ''
    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'name': opt['name'],
        'model': opt.get('network_g', {}).get('type', ''),
        'scale': opt.get('scale', ''),
        'dataset': dataset_name,
        'psnr': f"{metrics['psnr']:.4f}" if 'psnr' in metrics else '',
        'ssim': f"{metrics['ssim']:.4f}" if 'ssim' in metrics else '',
        'n_images': n_images,
        'runtime_s': f'{runtime_s:.1f}' if runtime_s is not None else '',
        'iter': checkpoint_iter(checkpoint),
        'checkpoint': checkpoint,
        'config': config_path,
        'git_commit': git[0],
        'git_dirty': git[1],
        'tag': tag,
    }
