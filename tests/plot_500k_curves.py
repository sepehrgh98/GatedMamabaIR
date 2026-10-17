"""Plot Set14 validation PSNR of the two x2 500k runs and their gap -> docs/500k_curves.png

    GMambaIR_env/bin/python tests/plot_500k_curves.py
"""
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

RUNS = [('MambaIRv2 (22.9M)', 'MambaIRv2_SR_x2_500k', '#2a78d6'),
        ('GatedMambaIR lean (25.1M)', 'GatedMambaIR_SR_x2_lean_chgate_500k', '#eb6834')]
INK, MUTED, GRID = '#1a1a19', '#5f5e5a', '#e6e5e0'


def psnr_curve(run):
    ea = EventAccumulator(f'tb_logger/{run}', size_guidance={'scalars': 0})
    ea.Reload()
    return {e.step: e.value for e in ea.Scalars('metrics/Set14/psnr')}


curves = {label: psnr_curve(run) for label, run, _ in RUNS}
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#8a8a85', 'axes.labelcolor': '#3d3d3a',
                     'xtick.color': MUTED, 'ytick.color': MUTED})
fig, (ax, gx) = plt.subplots(1, 2, figsize=(13, 5), facecolor='white')

for label, _, color in RUNS:
    steps = sorted(curves[label])
    x, y = np.array(steps) / 1000, [curves[label][s] for s in steps]
    ax.plot(x, y, color=color, lw=2, marker='o', ms=5, mec='white', mew=1.2, label=label)
    ax.annotate(f'{y[-1]:.2f} @ {x[-1]:.0f}k', (x[-1], y[-1]), xytext=(6, 0), textcoords='offset points',
                va='center', color='#3d3d3a', fontsize=10)
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
gx.text(0.02, 0.96, f'lean ahead at {ahead} of {len(gap)} checkpoints, mean {gap.mean():+.2f} dB',
        transform=gx.transAxes, va='top', color=INK, fontsize=10)
for a in (ax, gx):
    a.grid(axis='y', color=GRID, lw=0.8)
    a.title.set_color(INK)
fig.text(0.5, -0.02, 'x2 SR, 500k-step runs in progress, same recipe (batch 4, A100). '
         'Set14 has 14 images, so single checkpoints are noisy.', ha='center', color=MUTED, fontsize=10)
fig.tight_layout()
fig.savefig('docs/500k_curves.png', dpi=130, bbox_inches='tight')
print({label: (max(c), f'{max(c, key=c.get)} ', len(c)) for label, c in curves.items()}, 'gap points', len(gap))
