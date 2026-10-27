"""Size-dependent mask threshold on top of the v2 instances.

v2 draws every kept component (t=.4, closing 8, area >= 300, mean >= .85) at threshold .4. The
diagnosis showed small filaments come out ~2.4x too big and large ones ~half size, so here each kept
component is redrawn at its own threshold, picked per area bucket of its .4 component:
  t >= .4 : the component's pixels with prob >= t (shrinks it)
  t <  .4 : plus pixels with prob >= t connected to it inside its Voronoi cell (grows it)
Voronoi cells keep masks disjoint for any mix of thresholds. The kept set never changes, so only
matched IoUs (and which pairs pass .5) move.

Nested by date-grouped fold. Gate: +.005 PQ overall and better in >= 4/5 folds. If it passes, the
rule is refit on all folds and the fold 0+1 test ensemble (the 0.39 submission) is exported.

usage: SOLAR_DATA=<data dir> python analysis/levels.py <oof runs dir> <work dir>
"""
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from pycocotools import mask as mu
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_table as bt  # also puts src/ on sys.path
from fastsweep import components  # noqa: E402
from postproc import upsample  # noqa: E402

LEVELS = np.round(np.arange(.2, .71, .05), 2)
BASE = int(np.flatnonzero(LEVELS == .4)[0])
EDGES = np.array([600, 1000, 2000, 4000, 8000])  # px; every kept component is >= 300
OOF, WORK = Path(sys.argv[1]), Path(sys.argv[2])


def kept(prob):
    lab, f = components(prob, .4, 8)
    keep = (f[:, 0] >= 300) & (f[:, 1] >= .85)
    remap = np.zeros(len(f) + 1, np.int32)
    remap[1:][keep] = np.arange(1, keep.sum() + 1)
    return remap[lab], f[keep, 0]


def voronoi(lab):
    _, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
    return lab[iy, ix]


def redraw(prob, lab, cell, t):
    """Label map with every kept component redrawn at threshold t."""
    on = prob >= t * 255
    if t >= .4:
        return lab * on
    blob, _ = ndi.label(on, np.ones((3, 3)))
    key = blob.astype(np.int64) * (int(lab.max()) + 1) + cell
    own = np.zeros(int(key.max()) + 1, np.int32)
    own[key[lab > 0]] = lab[lab > 0]  # a (blob ∩ cell) piece joins the component it touches
    return np.where(on, own[key], 0)


def bucket(area):
    return np.searchsorted(EDGES, area, side='right')


def oof_row(args):
    _, fold, prob_u8, recs = args
    prob = upsample(prob_u8)
    lab, area = kept(prob)
    k = len(area)
    cell = voronoi(lab) if k else lab
    areas = np.zeros((len(LEVELS), k), np.int64)
    inter = [np.zeros((len(LEVELS), len(rec), k), np.int64) for rec in recs]
    for li, t in enumerate(LEVELS):
        flat = (redraw(prob, lab, cell, t) if k else lab).ravel()
        areas[li] = np.bincount(flat, minlength=k + 1)[1:]
        for r, rec in enumerate(recs):
            for g, ix in enumerate(rec):
                inter[r][li, g] = np.bincount(flat[ix], minlength=k + 1)[1:]
    return dict(fold=fold, area=area, areas=areas, inter=inter, gt=[np.array([len(ix) for ix in rec]) for rec in recs])


def ious(row):
    for inter, G in zip(row['inter'], row['gt']):
        yield inter / np.maximum(row['areas'][:, None, :] + G[None, :, None] - inter, 1)  # (L, n_gt, k)


def contributions(rows):
    """Per component and level: matched IoU sum, TP count, FP count over the image's expert records."""
    s, tp, fp = [], [], []
    for row in rows:
        k = len(row['area'])
        a, b, c = np.zeros((len(LEVELS), k)), np.zeros((len(LEVELS), k)), np.zeros((len(LEVELS), k))
        for iou in ious(row):
            hit = iou > .5
            a += (iou * hit).sum(1); b += hit.sum(1); c += ~hit.any(1)
        s.append(a); tp.append(b); fp.append(c)
    return np.concatenate(s, 1), np.concatenate(tp, 1), np.concatenate(fp, 1)


def pq(choice, mask, S, TP, FP, ngt):
    i = np.flatnonzero(mask)
    s, tp, fp = S[choice[i], i].sum(), TP[choice[i], i].sum(), FP[choice[i], i].sum()
    return s / (tp + .5 * fp + .5 * (ngt - tp))


def fit_rule(mask, B, S, TP, FP, ngt):
    rule = np.full(len(EDGES) + 1, BASE)
    for _ in range(3):  # coordinate ascent over buckets
        for b in range(len(rule)):
            def score(li):
                r = rule.copy(); r[b] = li
                return pq(r[B], mask, S, TP, FP, ngt)
            rule[b] = max(range(len(LEVELS)), key=score)
    return rule


def rubric(rows, rule):
    """Self-evaluation notebook statistics at the chosen levels."""
    pairs, g_deg, p_deg = [], [], []
    for row in rows:
        lv = rule[bucket(row['area'])]
        for iou in ious(row):
            iou = iou[lv, :, np.arange(len(lv))].T  # (n_gt, k) at each component's level
            pairs.append(iou[iou > 0]); g_deg += (iou > 0).sum(1).tolist(); p_deg += (iou > 0).sum(0).tolist()
    pairs, g_deg, p_deg = np.concatenate(pairs), np.array(g_deg), np.array(p_deg)
    return dict(mean_iou=round(pairs.mean(), 4), mean_dice=round((2 * pairs / (1 + pairs)).mean(), 4),
                gt_missed=int((g_deg == 0).sum()), gt_split=int((g_deg >= 2).sum()),
                pred_missed=int((p_deg == 0).sum()), pred_merged=int((p_deg >= 2).sum()))


def export(rule, out):
    zs = [np.load(OOF / 'v2' / f'probs_f{f}.npz') for f in (0, 1)]
    files = [str(x) for x in zs[0]['test_files']]
    assert all([str(x) for x in z['test_files']] == files for z in zs)
    rows = []
    for i, f in enumerate(files):
        prob = np.mean([upsample(z['test_prob'][i]).astype(np.float32) for z in zs], 0)  # as make_submission
        lab, area = kept(prob)
        if not len(area):
            continue
        cell, lv = voronoi(lab), LEVELS[rule[bucket(area)]]
        final = np.zeros_like(lab)
        for t in np.unique(lv):
            m = redraw(prob, lab, cell, t)
            sel = np.isin(m, np.flatnonzero(lv == t) + 1)
            assert not final[sel].any()
            final[sel] = m[sel]
        n, stem = 0, f.rsplit('.', 1)[0]
        for c in range(1, len(area) + 1):
            m = final == c
            if m.any():
                n += 1
                rows.append(f"{stem}_{n},{mu.encode(np.asfortranarray(m.astype(np.uint8)))['counts'].decode()}")
    out.write_text('filament_id,segmentation_rle\n' + '\n'.join(rows) + '\n')
    return len(rows)


if __name__ == '__main__':
    with get_context('fork').Pool(max(1, os.cpu_count() - 4)) as pool:
        rows = pool.map(oof_row, bt.tasks(OOF), chunksize=4)
    S, TP, FP = contributions(rows)
    cfold = np.concatenate([np.full(len(r['area']), r['fold']) for r in rows])
    B = bucket(np.concatenate([r['area'] for r in rows]))
    ngt = {k: sum(len(G) for r in rows if r['fold'] == k for G in r['gt']) for k in range(5)}
    base = np.full(len(B), BASE)
    assert abs(pq(base, cfold >= 0, S, TP, FP, sum(ngt.values())) - .44113363) < 1e-6, 'level .4 must reproduce v2'
    print('components per area bucket:', np.bincount(B).tolist(), '| levels:', LEVELS.tolist())
    cand = np.zeros(len(B), int)
    gains = []
    for k in range(5):
        train = cfold != k
        rule = fit_rule(train, B, S, TP, FP, sum(v for j, v in ngt.items() if j != k))
        cand[cfold == k] = rule[B][cfold == k]
        b0, c0 = pq(base, cfold == k, S, TP, FP, ngt[k]), pq(cand, cfold == k, S, TP, FP, ngt[k])
        gains.append(c0 - b0)
        print(f'fold {k}: v2 {b0:.4f}  size-levels {c0:.4f}  ({c0 - b0:+.4f})  rule {LEVELS[rule].tolist()}', flush=True)
    allm, n_all = cfold >= 0, sum(ngt.values())
    b0, c0 = pq(base, allm, S, TP, FP, n_all), pq(cand, allm, S, TP, FP, n_all)
    passed = c0 - b0 >= .005 and sum(g > 0 for g in gains) >= 4
    print(f'overall: v2 {b0:.4f}  size-levels (nested) {c0:.4f}  gain {c0 - b0:+.4f}  gate passed: {passed}')
    final_rule = fit_rule(allm, B, S, TP, FP, n_all)
    print('rule refit on all folds:', dict(zip(['<600', '600-1k', '1k-2k', '2k-4k', '4k-8k', '>8k'], LEVELS[final_rule].tolist())))
    print('rubric v2         :', rubric(rows, np.full(len(EDGES) + 1, BASE)))
    print('rubric size-levels:', rubric(rows, final_rule))
    (WORK / 'size_levels_rule.json').write_text(json.dumps(dict(
        edges_px=EDGES.tolist(), levels=LEVELS[final_rule].tolist(), nested_pq=c0, v2_pq=b0,
        fold_gains=gains, gate_passed=bool(passed)), indent=1))
    if passed:
        out = WORK / 'sub_v2_f01_size_levels.csv'
        print('exported', export(final_rule, out), 'rows to', out)
