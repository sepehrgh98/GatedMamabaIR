"""Automatic crop and query picking for analysis/memory_heatmap.py (numpy + OpenCV only, no model needed).

A good query for the motivation figure sits on a structure that repeats far away in the image (a window
on a facade). We score each candidate pixel by how many patches elsewhere in the image look like the patch
around it:

    score(p) = #{ r : corr(P_p, P_r) > --sim  and  ||p - r|| > --far }

where P is a mean-subtracted, contrast-normalised colour patch, so corr is a normalised cross-correlation
that ignores brightness / contrast changes between windows. Flat patches (sky, walls) are skipped, since
their correlation is just noise.

  * pick_crop:    the crop window with the highest total score (where the repetition is densest).
  * pick_queries: the n best-scoring pixels in the (cropped) image, at least min_sep pixels apart.

Run directly to preview the picks without a GPU:

    python analysis/heatmap_autopick.py datasets/SR/Urban100/LR_bicubic/X2/img035x2.png --out preview.png
"""
import argparse

import cv2
import numpy as np


def _patches(rgb, coords, half):
    """Normalised colour patches (len(coords), (2*half+1)^2*3) around coords, plus each patch's std."""
    pad = cv2.copyMakeBorder(rgb, half, half, half, half, cv2.BORDER_REFLECT)
    k = 2 * half + 1
    out = np.stack([pad[r:r + k, c:c + k].reshape(-1) for r, c in coords]).astype(np.float32)
    out -= out.mean(1, keepdims=True)
    std = out.std(1)
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
    return out, std


def repetition_score(rgb, cand_stride=4, ref_stride=2, half=8, sim=0.85, far=24, min_std=0.04, chunk=512):
    """Score map on the candidate grid. Returns (rows, cols, score[len(rows), len(cols)])."""
    h, w = rgb.shape[:2]
    rows, cols = np.arange(half, h - half, cand_stride), np.arange(half, w - half, cand_stride)
    cand = np.array([(r, c) for r in rows for c in cols])
    rr, rc = np.arange(half, h - half, ref_stride), np.arange(half, w - half, ref_stride)
    ref = np.array([(r, c) for r in rr for c in rc])
    pc, sc = _patches(rgb, cand, half)
    pr, sr = _patches(rgb, ref, half)
    keep = sr > min_std  # flat reference patches never count as a match
    pr, ref = pr[keep], ref[keep]
    score = np.zeros(len(cand), np.float32)
    for i in range(0, len(cand), chunk):
        corr = pc[i:i + chunk] @ pr.T
        d2 = ((cand[i:i + chunk, None, :] - ref[None, :, :]) ** 2).sum(-1)
        score[i:i + chunk] = ((corr > sim) & (d2 > far * far)).sum(1)
    score[sc <= min_std] = 0
    return rows, cols, score.reshape(len(rows), len(cols))


def pick_crop(rgb, size=256, stride=16, **kw):
    """(r0, c0, h, w) of the size x size window with the most repetition (smaller if the image is)."""
    h, w = rgb.shape[:2]
    ch, cw = min(size, h), min(size, w)
    if ch == h and cw == w:
        return 0, 0, h, w
    small = cv2.resize(rgb, (w // 2, h // 2), interpolation=cv2.INTER_AREA)  # coarse pass is enough here
    rows, cols, s = repetition_score(small, cand_stride=2, ref_stride=2, half=3, far=kw.get('far', 24) // 2)
    best, best_rc = -1.0, (0, 0)
    for r0 in range(0, h - ch + 1, stride):
        for c0 in range(0, w - cw + 1, stride):
            mr = (rows * 2 >= r0) & (rows * 2 < r0 + ch)
            mc = (cols * 2 >= c0) & (cols * 2 < c0 + cw)
            v = s[np.ix_(mr, mc)].sum()
            if v > best:
                best, best_rc = v, (r0, c0)
    return best_rc[0], best_rc[1], ch, cw


def snap_to_flat(rgb, r, c, radius=6, win=5):
    """Move (r, c) to the flattest pixel within +-radius (the inside of a pane rather than its frame)."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    var = cv2.blur(g * g, (win, win)) - cv2.blur(g, (win, win)) ** 2
    h, w = g.shape
    r0, r1, c0, c1 = max(r - radius, 0), min(r + radius + 1, h), max(c - radius, 0), min(c + radius + 1, w)
    i, j = np.unravel_index(np.argmin(var[r0:r1, c0:c1]), (r1 - r0, c1 - c0))
    return int(r0 + i), int(c0 + j)


def pick_queries(rgb, n=3, min_sep=48, border=24, snap=6, **kw):
    """n query points (row, col) on repeating structure, greedily, at least min_sep pixels apart.

    Candidates are scored with a patch big enough to hold a window and its frame, then snapped by up to
    snap pixels to the flattest nearby pixel, so the query lands inside a pane (as the hand-picked ones did).
    """
    rows, cols, s = repetition_score(rgb, **kw)
    h, w = rgb.shape[:2]
    order = np.argsort(-s, axis=None)
    picks = []
    for idx in order:
        i, j = np.unravel_index(idx, s.shape)
        if s[i, j] <= 0:
            break
        r, c = int(rows[i]), int(cols[j])
        if snap:
            r, c = snap_to_flat(rgb, r, c, snap)
        if not (border <= r < h - border and border <= c < w - border):
            continue
        if all((r - pr) ** 2 + (c - pc) ** 2 >= min_sep ** 2 for pr, pc in picks):
            picks.append((r, c))
        if len(picks) == n:
            break
    return picks or [(h // 2, w // 2)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('image')
    p.add_argument('--crop_size', type=int, default=256)
    p.add_argument('--n', type=int, default=3)
    p.add_argument('--out', default='autopick_preview.png')
    a = p.parse_args()
    rgb = cv2.cvtColor(cv2.imread(a.image), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    r0, c0, ch, cw = pick_crop(rgb, a.crop_size)
    crop = rgb[r0:r0 + ch, c0:c0 + cw]
    qs = pick_queries(crop, a.n)
    print(f'crop {r0},{c0},{ch},{cw}  queries {qs}')
    vis = (cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) * 255).astype(np.uint8)
    cv2.rectangle(vis, (c0, r0), (c0 + cw - 1, r0 + ch - 1), (0, 255, 255), 2)
    for r, c in qs:
        cv2.drawMarker(vis, (c0 + c, r0 + r), (0, 255, 0), cv2.MARKER_STAR, 14, 2)
    cv2.imwrite(a.out, vis)


if __name__ == '__main__':
    main()
