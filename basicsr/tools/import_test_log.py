"""Backfill results CSV from BasicSR test logs written before the tracker existed.

Usage:
    python basicsr/tools/import_test_log.py results/test_*/test_*.log [--csv docs/results.csv] [--tag baseline]
"""
import argparse
import importlib.util
import re
import sys
from datetime import datetime
from os import path as osp

# load results_tracker by path: importing the basicsr package pulls in torch and every arch
_spec = importlib.util.spec_from_file_location(
    'results_tracker', osp.join(osp.dirname(__file__), osp.pardir, 'utils', 'results_tracker.py'))
_tracker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tracker)
append_rows, checkpoint_iter = _tracker.append_rows, _tracker.checkpoint_iter

TIME_FMT = '%Y-%m-%d %H:%M:%S,%f'
STAMP = r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) INFO: '


def parse_log(log_path):
    with open(log_path) as f:
        text = f.read()

    def first(pattern, default=''):
        match = re.search(pattern, text, re.MULTILINE)
        return match.group(1).strip() if match else default

    name = first(r'^  name: (.+)$')
    scale = first(r'^  scale: (.+)$')
    model = first(r'^  network_g:\[\n    type: (.+)$')
    checkpoint = first(r'^    pretrain_network_g: (.+)$')
    n_images = dict(re.findall(r'Number of test images in (\S+): (\d+)', text))

    starts = {m.group(2): m.group(1) for m in re.finditer(STAMP + r'Testing (\S+)\.\.\.$', text, re.MULTILINE)}
    rows = []
    for match in re.finditer(STAMP + r'Validation (\S+)\n((?:\s+# .+\n?)+)', text, re.MULTILINE):
        end_time, dataset, block = match.groups()
        metrics = dict(re.findall(r'# (\w+): ([\d.]+)', block))
        runtime = ''
        if dataset in starts:
            delta = datetime.strptime(end_time, TIME_FMT) - datetime.strptime(starts[dataset], TIME_FMT)
            runtime = f'{delta.total_seconds():.1f}'
        rows.append({
            'timestamp': end_time.split(',')[0],
            'name': name,
            'model': model,
            'scale': scale,
            'dataset': dataset,
            'psnr': metrics.get('psnr', ''),
            'ssim': metrics.get('ssim', ''),
            'n_images': n_images.get(dataset, ''),
            'runtime_s': runtime,
            'iter': checkpoint_iter(checkpoint),
            'checkpoint': checkpoint,
            'config': '',
            'git_commit': 'unknown',
            'git_dirty': '',
        })
    return rows


def main():
    root = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir, osp.pardir))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('logs', nargs='+', help='BasicSR test log files')
    parser.add_argument('--csv', default=osp.join(root, 'docs', 'results.csv'))
    parser.add_argument('--tag', default='imported', help='value for the tag column')
    args = parser.parse_args()

    for log_path in args.logs:
        rows = parse_log(log_path)
        for row in rows:
            row['tag'] = args.tag
        if rows:
            append_rows(args.csv, rows)
        print(f'{log_path}: {len(rows)} rows -> {args.csv}', file=sys.stderr if not rows else sys.stdout)


if __name__ == '__main__':
    main()
