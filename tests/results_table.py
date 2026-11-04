"""Markdown PSNR / SSIM tables (x2, x3, x4) for docs/BASELINES.md from docs/results.csv.

Rows: the same-setup baselines (tag baseline-sameSetup-<Model>), MambaIRv2 and DeltaSR (tags main500k-*, main-ft250k-*).
For each test config the newest row per dataset wins (e.g. SAN's chop re-test replaces the whole-image OOM run).
RCAN x4 uses only the gradient-clipping run (test_RCAN_SR_x4_ft250k_gclip).
usage: python tests/results_table.py [path/to/results.csv]
"""
import csv
import sys

CSV = sys.argv[1] if len(sys.argv) > 1 else 'docs/results.csv'
SETS = ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']
# display order: weakest-era first, ours last
ORDER = ['EDSR', 'RCAN', 'SAN', 'SwinIR', 'HAT', 'DAT', 'CAT_A', 'MambaIR', 'MambaIRv2', 'DeltaSR']
NAMES = {'CAT_A': 'CAT-A', 'DeltaSR': '**DeltaSR (ours)**'}
OURS = {'main500k-lean', 'main-ft250k-lean'}
REF = {'main500k-mambairv2', 'main-ft250k-mambairv2'}

best = {}  # (model, scale, dataset) -> (timestamp, psnr, ssim)
for r in csv.DictReader(open(CSV)):
    tag, name = r['tag'], r['name']
    if tag.startswith('baseline-sameSetup-'):
        model = tag[len('baseline-sameSetup-'):]
        if model == 'RCAN' and name == 'test_RCAN_SR_x4_ft250k':
            continue  # diverged run, replaced by the _gclip run
    elif tag in OURS:
        model = 'DeltaSR'
    elif tag in REF:
        model = 'MambaIRv2'
    else:
        continue
    if not r['psnr'] or not r['ssim']:
        continue
    key = (model, int(r['scale']), r['dataset'])
    if key not in best or r['timestamp'] > best[key][0]:
        best[key] = (r['timestamp'], float(r['psnr']), float(r['ssim']))

for scale in (2, 3, 4):
    rows = [m for m in ORDER if any((m, scale, d) in best for d in SETS)]
    if not rows:
        continue
    # best PSNR per dataset, bolded
    top = {d: max((best[(m, scale, d)][1] for m in rows if (m, scale, d) in best), default=None) for d in SETS}
    print(f'**×{scale}** (PSNR / SSIM)\n')
    print('| model | ' + ' | '.join(SETS) + ' |')
    print('|---|' + '---|' * len(SETS))
    for m in rows:
        cells = []
        for d in SETS:
            if (m, scale, d) not in best:
                cells.append('–')
                continue
            _, p, s = best[(m, scale, d)]
            c = f'{p:.2f} / {s:.4f}'
            cells.append(f'**{c}**' if top[d] is not None and round(p, 2) == round(top[d], 2) else c)
        print(f'| {NAMES.get(m, m)} | ' + ' | '.join(cells) + ' |')
    print()
