"""Plot Set14 validation PSNR of MambaIRv2 vs lean GatedMambaIR, their gap, and the final test gain on all 5 sets
-> docs/500k_curves.png (x2), docs/x3_curves.png, docs/x4_curves.png

    GMambaIR_env/bin/python tests/plot_curves.py --scale 2   # or 3, 4

x3/x4 are 250k fine-tunes from the x2 500k weights. If a run has no test rows in docs/results.csv yet,
the right panel says so; rerun once the tests finish.
"""
import argparse
import csv

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

# scale -> (runs, total steps, output, footnote)
CONFIGS = {
    2: ([('MambaIRv2 (22.9M)', 'MambaIRv2_SR_x2_500k', '#2a78d6'),
         ('GatedMambaIR lean (25.1M)', 'GatedMambaIR_SR_x2_lean_chgate_500k', '#eb6834')],
        500000, 'docs/500k_curves.png',
        'x2 SR, both runs finished 500k steps with the same recipe (batch 4, A100).'),
    3: ([('MambaIRv2', 'MambaIRv2_SR_x3_ft250k', '#2a78d6'),
         ('GatedMambaIR lean', 'GatedMambaIR_SR_x3_lean_chgate_ft250k', '#eb6834')],
        250000, 'docs/x3_curves.png',
        'x3 SR, both runs fine-tuned 250k steps from their x2 500k weights (paper schedule, lr 1e-4).'),
    4: ([('MambaIRv2', 'MambaIRv2_SR_x4_ft250k', '#2a78d6'),
         ('GatedMambaIR lean', 'GatedMambaIR_SR_x4_lean_chgate_ft250k', '#eb6834')],
        250000, 'docs/x4_curves.png',
        'x4 SR, both runs fine-tuned 250k steps from their x2 500k weights (paper schedule, lr 1e-4).'),
}
INK, MUTED, GRID = '#1a1a19', '#5f5e5a', '#e6e5e0'
SETS = ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']

parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=int, default=2, choices=sorted(CONFIGS))
args = parser.parse_args()
RUNS, TOTAL, OUT, NOTE = CONFIGS[args.scale]
LATE = TOTAL // 5  # "since 100k" for x2, "since 50k" for the fine-tunes


def psnr_curve(run):
    ea = EventAccumulator(f'tb_logger/{run}', size_guidance={'scalars': 0})
    ea.Reload()
    # drop the extra end-of-training validation (step TOTAL+2), it repeats the last one
    return {e.step: e.value for e in ea.Scalars('metrics/Set14/psnr') if e.step <= TOTAL}


def final_psnr(run):
    with open('docs/results.csv') as f:
        return {r['dataset']: float(r['psnr']) for r in csv.DictReader(f) if r['name'] == f'test_{run}'}


curves = {label: psnr_curve(run) for label, run, _ in RUNS}
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#8a8a85', 'axes.labelcolor': '#3d3d3a',
                     'xtick.color': MUTED, 'ytick.color': MUTED})
fig, (ax, gx, tx) = plt.subplots(1, 3, figsize=(18, 5), facecolor='white',
                                 gridspec_kw={'width_ratios': [1.15, 1, 0.85]})

for label, _, color in RUNS:
    steps = sorted(curves[label])
    x, y = np.array(steps) / 1000, [curves[label][s] for s in steps]
    ax.plot(x, y, color=color, lw=2, marker='o', ms=5, mec='white', mew=1.2,
            label=f'{label}: {y[-1]:.2f} @ {x[-1]:.0f}k')
# start the y axis just under the earliest checkpoints so the x2 warm-up does not squash the rest
ylo = min(np.percentile(list(c.values()), 3) for c in curves.values()) - 0.05
ax.set(title='Set14 validation PSNR (higher is better)', xlabel='training steps (k)', ylabel='PSNR (dB)',
       ylim=(32.8 if args.scale == 2 else ylo, None), xlim=(0, TOTAL / 1000 * 1.02))
ax.legend(frameon=False, loc='lower right')

base, lean = curves[RUNS[0][0]], curves[RUNS[1][0]]
common = sorted(set(base) & set(lean))
gap = np.array([lean[s] - base[s] for s in common])
gx.axhline(0, color='#8a8a85', lw=1)
gx.bar(np.array(common) / 1000, gap, width=3.6, color=['#eb6834' if g >= 0 else '#2a78d6' for g in gap])
gx.set(title='Gap: lean minus MambaIRv2 (above 0 = lean ahead)', xlabel='training steps (k)', ylabel='dB',
       xlim=(0, TOTAL / 1000 * 1.02))
ahead = int((gap > 0).sum())
late = np.array(common) >= LATE
gx.set_ylim(min(gap.min(), -0.1) - 0.05, gap.max() + 0.18)
gx.text(0.02, 0.97, f'lean ahead at {ahead} of {len(gap)} checkpoints (mean {gap.mean():+.2f} dB)\n'
        f'since {LATE // 1000}k: {int((gap[late] > 0).sum())} of {int(late.sum())} (mean {gap[late].mean():+.2f} dB)',
        transform=gx.transAxes, va='top', color=INK, fontsize=10,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))
base_t, lean_t = (final_psnr(run) for _, run, _ in RUNS)
if all(d in base_t and d in lean_t for d in SETS):
    tgap = np.array([lean_t[d] - base_t[d] for d in SETS])
    tx.axhline(0, color='#8a8a85', lw=1)
    tx.bar([f'{d}\n{lean_t[d]:.2f}\nvs {base_t[d]:.2f}' for d in SETS], tgap, width=0.6,
           color=['#eb6834' if g >= 0 else '#2a78d6' for g in tgap])
    for i, d in enumerate(SETS):
        tx.annotate(f'{tgap[i]:+.2f}', (i, tgap[i]), xytext=(0, 4 if tgap[i] >= 0 else -4),
                    textcoords='offset points', ha='center', va='bottom' if tgap[i] >= 0 else 'top',
                    color=INK, fontsize=10, weight='bold')
    tx.set_ylim(min(tgap.min(), 0) - 0.02, tgap.max() + 0.04)
    tests_done = True
else:
    tx.set_xticks(range(len(SETS)), SETS)
    tx.set_yticks([])
    tx.text(0.5, 0.5, 'final test pending\n(no lean rows in results.csv yet)', transform=tx.transAxes,
            ha='center', va='center', color=MUTED, fontsize=12)
    tests_done = False
tx.set(title=f'Final test at {TOTAL // 1000}k: lean minus MambaIRv2', ylabel='dB')
for a in (ax, gx, tx):
    a.grid(axis='y', color=GRID, lw=0.8)
    a.title.set_color(INK)
fig.text(0.5, -0.02, f'{NOTE} Left/middle: Set14 validation (14 images, noisy per checkpoint). '
         'Right: full test sets, final model.', ha='center', color=MUTED, fontsize=10)
fig.tight_layout()
fig.savefig(OUT, dpi=130, bbox_inches='tight')
print(OUT, {label: (max(c.values()), max(c, key=c.get), len(c)) for label, c in curves.items()},
      'gap points', len(gap), 'tests done', tests_done)
