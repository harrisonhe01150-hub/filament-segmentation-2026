"""Decision layer: choose filament instances from a probability map with a component tree.

A fixed threshold answers "which pixels" and "which blobs" with one number for the whole image. Here every
connected group of the map at every threshold level is a candidate (a node). Nodes of a higher level lie inside
nodes of a lower level, so the candidates form a tree, and any set of nodes with no ancestor/descendant pair
is a valid submission (disjoint masks). A learned scorer estimates, for each node, the IoU it would earn against
an annotator (0 if it would not match), and `select` picks the set of nodes with the largest total of
(score - tau) exactly, by dynamic programming over the tree. The same recursion decides, per filament, how far
the mask extends (which level) and whether touching pieces are one filament or several.

Why (score - tau): PQ = sum(matched IoU) / (TP + FP/2 + FN/2) is a ratio; adding a prediction raises it exactly
when its expected matched IoU exceeds PQ / 2 (Dinkelbach), so one global tau is the right inclusion rule.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi

LEVELS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80)
CLOSE = 8        # same grouping radius as the fixed-rule pipeline (predict.POST)
MIN_AREA = 60    # px at 2048; smaller nodes are never candidates
CENTRE, RADIUS = 1023.5, 900.0
FEATURES = (
    'level', 'area', 'log_area', 'p_mean', 'p_std', 'p_max', 'frac_up1', 'frac_up2', 'frac_up4',
    'cx_rho', 'elong', 'minor', 'major', 'bbox_fill', 'compact', 'n_frag', 'frag_share',
    'img_mean', 'img_contrast', 'disagree', 'p_mean_min', 'parent_ratio', 'parent_p_mean', 'n_child',
    'child_ratio', 'peak_gap', 'img_nodes40', 'img_pmean', 'rank_in_img',
)
COL = {name: i for i, name in enumerate(FEATURES)}


def _kernel(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def level_labels(prob, t, close=CLOSE):
    """Groups at threshold t (prob on the 0-255 scale): pixels >= t, joined by a closing of radius `close`.
    Returns (label map, number of labels, the thresholded mask)."""
    lo = prob >= t * 255
    grp = cv2.morphologyEx(lo.astype(np.uint8), cv2.MORPH_CLOSE, _kernel(close)) if close else lo.astype(np.uint8)
    n, lab = cv2.connectedComponents(grp, connectivity=8, ltype=cv2.CV_32S)   # same partition as scipy's 8-connected label
    return lab * lo, n - 1, lo


def _one_pixel(flat, n):
    """Index of one pixel of every label 1..n (0 for labels without pixels)."""
    first = np.zeros(n + 1, np.int64)
    pos = np.flatnonzero(flat)
    first[flat[pos][::-1]] = pos[::-1]
    return first


def build_tree(prob, image=None, members=None, levels=LEVELS, close=CLOSE, min_area=MIN_AREA, keep_labels=False):
    """prob: float32 (2048, 2048) on the 0-255 scale. image: uint8 gray frame (contrast features).
    members: optional list of per-model maps on the same scale (disagreement features).

    Returns dict with nodes (N, len(FEATURES)) float32, level (N,) index into `levels`, comp (N,) label of the
    node inside its level, parent (N,) node index or -1, and labels (one label map per level) if keep_labels.
    Nodes are ordered by level, so a parent always comes before its children.
    """
    H, W = prob.shape
    sel = np.flatnonzero(prob.ravel() >= levels[0] * 255)   # every node lives inside the lowest level (~1% of the frame)
    ys, xs = np.divmod(sel, W)
    xr, yr = (xs - CENTRE).astype(np.float64), (ys - CENTRE).astype(np.float64)
    pf = prob.ravel()[sel].astype(np.float64)
    imf = contrast = None
    if image is not None:
        im = image.astype(np.float32)
        contrast = (im - cv2.GaussianBlur(im, (0, 0), 12)).ravel()[sel].astype(np.float64)
        imf = im.ravel()[sel].astype(np.float64)
    dis = mem = None
    if members is not None and len(members) > 1:
        mem = [m.ravel()[sel].astype(np.float64) for m in members]
        dis = np.max(mem, 0) - np.min(mem, 0)
    masks = [(pf >= t * 255).astype(np.float64) for t in levels]
    rows, lev, comp, parent, labels = [], [], [], [], []
    prev_flat = prev_index = None
    offset = 0
    for k, t in enumerate(levels):
        lab, n, lo = level_labels(prob, t, close)
        flat = lab.ravel()
        area = np.bincount(flat, minlength=n + 1).astype(np.float64)
        ok = area >= min_area
        ok[0] = False
        ids = np.flatnonzero(ok)
        index = np.full(n + 1, -1, np.int64)
        index[ids] = offset + np.arange(len(ids))
        if keep_labels:
            labels.append(lab)
        if len(ids) == 0:
            prev_flat, prev_index = flat, index
            continue

        fs = flat[sel]

        def s(w):  # per-node sum of a per-pixel quantity (w is given on the `sel` pixels)
            return np.bincount(fs, weights=w, minlength=n + 1)[ids]

        a = area[ids]
        sp, sp2 = s(pf), s(pf * pf)
        mean = sp / a
        std = np.sqrt(np.maximum(sp2 / a - mean ** 2, 0))
        pm = np.zeros(n + 1)
        np.maximum.at(pm, fs, pf)
        pmax = pm[ids]
        ups = [s(masks[min(k + d, len(levels) - 1)]) / a for d in (1, 2, 4)]
        mx, my = s(xr) / a, s(yr) / a
        cxx, cyy, cxy = s(xr * xr) / a - mx ** 2, s(yr * yr) / a - my ** 2, s(xr * yr) / a - mx * my
        tr, det = cxx + cyy, cxx * cyy - cxy ** 2
        root = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
        l1, l2 = np.maximum(tr / 2 + root, 1e-6), np.maximum(tr / 2 - root, 1e-6)
        obj = ndi.find_objects(lab)
        bbox = np.array([(obj[i - 1][0].stop - obj[i - 1][0].start) * (obj[i - 1][1].stop - obj[i - 1][1].start)
                         for i in ids], np.float64)
        edge = lo & ~cv2.erode(lo.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        perim = np.maximum(s(edge.ravel()[sel].astype(np.float64)), 1)
        nf, frag = cv2.connectedComponents(lo.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S)  # pieces before grouping
        nf -= 1
        fr = frag.ravel()
        fa = np.bincount(fr, minlength=nf + 1)
        grp_of = flat[_one_pixel(fr, nf)]
        nfrag = np.bincount(grp_of[1:], minlength=n + 1)[ids].astype(np.float64)
        big = np.zeros(n + 1)
        np.maximum.at(big, grp_of[1:], fa[1:])
        if prev_flat is not None:
            par = prev_index[prev_flat[_one_pixel(flat, n)[ids]]]
        else:
            par = np.full(len(ids), -1, np.int64)
        f = np.zeros((len(ids), len(FEATURES)), np.float32)
        f[:, COL['level']] = t
        f[:, COL['area']] = a
        f[:, COL['log_area']] = np.log(a)
        f[:, COL['p_mean']] = mean / 255
        f[:, COL['p_std']] = std / 255
        f[:, COL['p_max']] = pmax / 255
        f[:, COL['frac_up1']], f[:, COL['frac_up2']], f[:, COL['frac_up4']] = ups
        f[:, COL['cx_rho']] = np.hypot(mx, my) / RADIUS
        f[:, COL['elong']] = np.sqrt(l1 / l2)
        f[:, COL['minor']], f[:, COL['major']] = 4 * np.sqrt(l2), 4 * np.sqrt(l1)
        f[:, COL['bbox_fill']] = a / np.maximum(bbox, 1)
        f[:, COL['compact']] = 4 * np.pi * a / perim ** 2
        f[:, COL['n_frag']] = nfrag
        f[:, COL['frag_share']] = big[ids] / a
        if imf is not None:
            f[:, COL['img_mean']] = s(imf) / a
            f[:, COL['img_contrast']] = s(contrast) / a
        if dis is not None:
            f[:, COL['disagree']] = s(dis) / a / 255
            f[:, COL['p_mean_min']] = np.min([s(m) / a for m in mem], 0) / 255
        else:
            f[:, COL['p_mean_min']] = mean / 255
        f[:, COL['peak_gap']] = pmax / 255 - t
        rows.append(f); lev.append(np.full(len(ids), k)); comp.append(ids); parent.append(par)
        prev_flat, prev_index = flat, index
        offset += len(ids)
    if not rows:
        z = np.zeros(0, np.int64)
        return dict(nodes=np.zeros((0, len(FEATURES)), np.float32), level=z, comp=z, parent=z, labels=labels)
    nodes = np.concatenate(rows)
    lev, comp, parent = np.concatenate(lev), np.concatenate(comp), np.concatenate(parent)
    has = parent >= 0
    nodes[has, COL['parent_ratio']] = nodes[has, COL['area']] / nodes[parent[has], COL['area']]
    nodes[has, COL['parent_p_mean']] = nodes[parent[has], COL['p_mean']]
    nodes[~has, COL['parent_ratio']] = 1
    nodes[~has, COL['parent_p_mean']] = nodes[~has, COL['p_mean']]
    nodes[:, COL['n_child']] = np.bincount(parent[has], minlength=len(nodes))
    nodes[:, COL['child_ratio']] = (np.bincount(parent[has], weights=nodes[has, COL['area']], minlength=len(nodes))
                                    / nodes[:, COL['area']])
    k40 = min(range(len(levels)), key=lambda i: abs(levels[i] - .4))
    nodes[:, COL['img_nodes40']] = int((lev == k40).sum())
    nodes[:, COL['img_pmean']] = float(prob.mean()) / 255
    order = np.argsort(-nodes[:, COL['p_mean']])
    rank = np.empty(len(nodes))
    rank[order] = np.arange(len(nodes))
    nodes[:, COL['rank_in_img']] = rank / max(len(nodes), 1)
    return dict(nodes=nodes, level=lev, comp=comp, parent=parent, labels=labels)


def select(score, parent, tau):
    """Exact maximiser of sum(score - tau) over sets of nodes without an ancestor/descendant pair.
    Nodes must be ordered with parents before children (the order build_tree returns).
    Returns the chosen node indices."""
    n = len(score)
    gain = np.asarray(score, np.float64) - tau
    below = np.zeros(n)                       # best total obtainable strictly inside each node
    best = np.zeros(n)
    for i in range(n - 1, -1, -1):
        best[i] = max(gain[i], below[i], 0.0)
        if parent[i] >= 0:
            below[parent[i]] += best[i]
    take = np.zeros(n, bool)
    blocked = np.zeros(n, bool)               # an ancestor was taken, or an ancestor's subtree is worthless
    for i in range(n):
        p = parent[i]
        if p >= 0 and (blocked[p] or take[p]):
            blocked[i] = True
        elif gain[i] > 0 and gain[i] >= below[i]:
            take[i] = True
        elif below[i] <= 0:
            blocked[i] = True                 # nothing worth taking inside
    return np.flatnonzero(take)


def render(prob, level, comp, chosen, levels=LEVELS, close=CLOSE):
    """Label map (0 = background, 1..K = instances) of the chosen nodes; K follows the order of `chosen`."""
    out = np.zeros(prob.shape, np.int32)
    number = {int(i): n + 1 for n, i in enumerate(chosen)}
    for k in sorted(set(level[chosen].tolist())):
        lab, _, _ = level_labels(prob, levels[k], close)
        for i in chosen[level[chosen] == k]:
            out[lab == comp[i]] = number[int(i)]
    return out
