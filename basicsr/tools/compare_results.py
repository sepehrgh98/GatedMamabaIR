"""Compare experiments recorded in the results CSV (written by basicsr/test.py).

Each experiment is identified as ``name@iter`` (or just ``name`` when the checkpoint has no
iteration). If the same experiment was tested on a dataset more than once, the newest row wins.

Examples:
    python basicsr/tools/compare_results.py --list
    python basicsr/tools/compare_results.py --scale 2
    python basicsr/tools/compare_results.py --scale 2 --metric both --baseline MambaIRv2
    python basicsr/tools/compare_results.py --scale 2 --baseline MambaIRv2 --markdown
"""
import argparse
import csv
import re
import sys
from os import path as osp

DATASET_ORDER = ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']


def load(csv_path):
    if not osp.exists(csv_path):
        sys.exit(f'No results file at {csv_path}. Run basicsr/test.py or basicsr/tools/import_test_log.py first.')
    with open(csv_path, newline='') as f:
        return list(csv.DictReader(f))


def exp_key(row):
    return f"{row['name']}@{row['iter']}" if row['iter'] else row['name']


def latest_per_dataset(rows):
    """{exp_key: {dataset: row}}, keeping the newest row for each pair, in first-seen experiment order."""
    table = {}
    for row in sorted(rows, key=lambda r: r['timestamp']):
        table.setdefault(exp_key(row), {})[row['dataset']] = row
    return table


def ordered_datasets(table):
    seen = {d for per_ds in table.values() for d in per_ds}
    return [d for d in DATASET_ORDER if d in seen] + sorted(seen - set(DATASET_ORDER))


def fmt_cell(row, base_row, metric):
    if row is None:
        return '-'
    parts = []
    for m, digits in (('psnr', 2), ('ssim', 4)):
        if metric not in (m, 'both') or not row.get(m):
            continue
        text = f'{float(row[m]):.{digits}f}'
        if base_row is not None and base_row.get(m) and base_row is not row:
            delta = float(row[m]) - float(base_row[m])
            text += f' ({delta:+.{digits}f})'
        parts.append(text)
    return ' / '.join(parts) or '-'


def render(header, body, markdown):
    if markdown:
        lines = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join('---' for _ in header) + ' |']
        lines += ['| ' + ' | '.join(r) + ' |' for r in body]
        return '\n'.join(lines)
    widths = [max(len(str(c)) for c in col) for col in zip(header, *body)]
    fmt = '  '.join(f'{{:<{w}}}' for w in widths)
    lines = [fmt.format(*header), fmt.format(*['-' * w for w in widths])]
    lines += [fmt.format(*r) for r in body]
    return '\n'.join(lines)


def main():
    root = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir, osp.pardir))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv', default=osp.join(root, 'docs', 'results.csv'))
    parser.add_argument('--scale', help='keep only this scale, e.g. 2')
    parser.add_argument('--name', help='regex on experiment name@iter')
    parser.add_argument('--tag', help='keep only rows with this tag')
    parser.add_argument('--datasets', nargs='+', help='datasets to show (default: all tested)')
    parser.add_argument('--metric', choices=['psnr', 'ssim', 'both'], default='psnr')
    parser.add_argument('--baseline', help='regex matching one experiment; adds deltas against it')
    parser.add_argument('--markdown', action='store_true', help='print a Markdown table')
    parser.add_argument('--list', action='store_true', help='list experiments and their metadata')
    args = parser.parse_args()

    rows = load(args.csv)
    if args.scale:
        rows = [r for r in rows if r['scale'] == args.scale]
    if args.tag:
        rows = [r for r in rows if r['tag'] == args.tag]
    if args.name:
        rows = [r for r in rows if re.search(args.name, exp_key(r))]
    if args.datasets:
        rows = [r for r in rows if r['dataset'] in args.datasets]
    if not rows:
        sys.exit('No rows match the filters.')

    table = latest_per_dataset(rows)

    if args.list:
        header = ['experiment', 'model', 'scale', 'datasets', 'runtime (min)', 'commit', 'tag', 'last tested']
        body = []
        for key, per_ds in table.items():
            any_row = max(per_ds.values(), key=lambda r: r['timestamp'])
            runtime = sum(float(r['runtime_s']) for r in per_ds.values() if r['runtime_s'])
            commit = any_row['git_commit'] + ('*' if any_row['git_dirty'] == 'yes' else '')
            body.append([key, any_row['model'], any_row['scale'], str(len(per_ds)), f'{runtime / 60:.1f}', commit,
                         any_row['tag'], any_row['timestamp']])
        print(render(header, body, args.markdown))
        return

    base = None
    if args.baseline:
        matches = [k for k in table if re.search(args.baseline, k)]
        if len(matches) != 1:
            sys.exit(f'--baseline must match exactly one experiment, matched: {matches or "none"}')
        base = table[matches[0]]

    datasets = ordered_datasets(table)
    header = ['experiment'] + datasets
    body = [[key] + [fmt_cell(per_ds.get(d), base.get(d) if base else None, args.metric) for d in datasets]
            for key, per_ds in table.items()]
    print(render(header, body, args.markdown))


if __name__ == '__main__':
    main()
