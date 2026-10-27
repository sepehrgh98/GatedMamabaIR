"""Plot Set14 validation PSNR of the two x2 500k runs, their gap, and the final test gain on all 5 sets
-> docs/500k_curves.png

    GMambaIR_env/bin/python tests/plot_500k_curves.py
"""
import csv

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

RUNS = [('MambaIRv2 (22.9M)', 'MambaIRv2_SR_x2_500k', '#2a78d6'),
        ('GatedMambaIR lean (25.1M)', 'GatedMambaIR_SR_x2_lean_chgate_500k', '#eb6834')]
INK, MUTED, GRID = '#1a1a19', '#5f5e5a', '#e6e5e0'
SETS = ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']


def psnr_curve(run):
    ea = EventAccumulator(f'tb_logger/{run}', size_guidance={'scalars': 0})
    ea.Reload()
    # drop the extra end-of-training validation (step 500002), it repeats 500k
    return {e.step: e.value for e in ea.Scalars('metrics/Set14/psnr') if e.step <= 500000}


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
ax.set(title='Set14 validation PSNR (higher is better)', xlabel='training steps (k)', ylabel='PSNR (dB)',
       ylim=(32.8, None))
ax.legend(frameon=False, loc='lower right')

base, lean = curves[RUNS[0][0]], curves[RUNS[1][0]]
common = sorted(set(base) & set(lean))
gap = np.array([lean[s] - base[s] for s in common])
gx.axhline(0, color='#8a8a85', lw=1)
gx.bar(np.array(common) / 1000, gap, width=3.6, color=['#eb6834' if g >= 0 else '#2a78d6' for g in gap])
gx.set(title='Gap: lean minus MambaIRv2 (above 0 = lean ahead)', xlabel='training steps (k)', ylabel='dB')
ahead = int((gap > 0).sum())
late = np.array(common) >= 100000
gx.set_ylim(min(gap.min(), -0.1) - 0.05, gap.max() + 0.18)
gx.text(0.02, 0.97, f'lean ahead at {ahead} of {len(gap)} checkpoints (mean {gap.mean():+.2f} dB)\n'
        f'since 100k: {int((gap[late] > 0).sum())} of {int(late.sum())} (mean {gap[late].mean():+.2f} dB)',
        transform=gx.transAxes, va='top', color=INK, fontsize=10,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))
base_t, lean_t = (final_psnr(run) for _, run, _ in RUNS)
tgap = np.array([lean_t[d] - base_t[d] for d in SETS])
tx.axhline(0, color='#8a8a85', lw=1)
tx.bar([f'{d}\n{lean_t[d]:.2f}\nvs {base_t[d]:.2f}' for d in SETS], tgap, width=0.6, color=['#eb6834' if g >= 0 else '#2a78d6' for g in tgap])
for i, d in enumerate(SETS):
    tx.annotate(f'{tgap[i]:+.2f}', (i, tgap[i]), xytext=(0, 4),
                textcoords='offset points', ha='center', va='bottom', color=INK, fontsize=10, weight='bold')
tx.set(title='Final test at 500k: lean minus MambaIRv2', ylabel='dB', ylim=(min(tgap.min(), 0) - 0.02, tgap.max() + 0.04))
for a in (ax, gx, tx):
    a.grid(axis='y', color=GRID, lw=0.8)
    a.title.set_color(INK)
fig.text(0.5, -0.02, 'x2 SR, both runs finished 500k steps with the same recipe (batch 4, A100). '
         'Left/middle: Set14 validation (14 images, noisy per checkpoint). Right: full test sets, final model.', ha='center', color=MUTED, fontsize=10)
fig.tight_layout()
fig.savefig('docs/500k_curves.png', dpi=130, bbox_inches='tight')
print({label: (max(c.values()), max(c, key=c.get), len(c)) for label, c in curves.items()}, 'gap points', len(gap))
