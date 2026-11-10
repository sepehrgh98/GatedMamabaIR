"""Contact sheet of memory_heatmap.py results, to pick the examples for the motivation figure.

Collects every side-by-side figure (<image>_q<r>_<c>_L<g.l>.png: input | model A | model B) under a results
folder and lays them out in a multi-page PDF, several per page, each with a running number and its tag.
The tag is the file-name prefix of the single panels (<tag>_input.png, <tag>_a.png, <tag>_b.png) used in LaTeX.

    python analysis/heatmap_sheet.py results/memory_heatmap/pass5 --layer 2.3 --out pass5_sheet.pdf
"""
import argparse
import glob
import os
import re
from os import path as osp

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

TAG = re.compile(r'_q\d+_\d+_L(\d+\.\d+)\.png$')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('root', help='results folder (searched recursively)')
    p.add_argument('--layer', default=None, help='only this layer, e.g. 2.3')
    p.add_argument('--per_page', type=int, default=4)
    p.add_argument('--out', default=None, help='PDF path (default <root>/sheet[_L<layer>].pdf)')
    a = p.parse_args()

    files = []
    for f in sorted(glob.glob(osp.join(a.root, '**', '*.png'), recursive=True)):
        m = TAG.search(osp.basename(f))
        if m and (a.layer is None or m.group(1) == a.layer):
            files.append(f)
    assert files, f'no side-by-side figures under {a.root}'
    out = a.out or osp.join(a.root, f'sheet{"_L" + a.layer if a.layer else ""}.pdf')

    with PdfPages(out) as pdf, open(osp.splitext(out)[0] + '_index.txt', 'w') as idx:
        for start in range(0, len(files), a.per_page):
            chunk = files[start:start + a.per_page]
            fig, axes = plt.subplots(a.per_page, 1, figsize=(11, 3.9 * a.per_page))
            for k, ax in enumerate(axes):
                ax.set_axis_off()
                if k < len(chunk):
                    n, tag = start + k + 1, osp.basename(chunk[k])[:-4]
                    ax.imshow(plt.imread(chunk[k]))
                    ax.set_title(f'#{n}   {tag}', fontsize=9, loc='left')
                    idx.write(f'{n}\t{tag}\t{osp.relpath(chunk[k], a.root)}\n')
            fig.tight_layout()
            pdf.savefig(fig, dpi=110)
            plt.close(fig)
    print(f'{len(files)} examples -> {out}')


if __name__ == '__main__':
    main()
