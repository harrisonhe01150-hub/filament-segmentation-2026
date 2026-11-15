"""K1 re-implementation, step 1: stage-1 seeds, per-seed IoU against every annotator record, and the
fixed keep rule (area >= 300, mean probability >= 0.85) scored on real label maps with postproc.score_file.

Inputs used: OOF probability maps of the two model sets, scratch/bank/gt_index.pkl (TRAIN annotations only).
Writes scratch/ws/k1_reimpl/seeds_f<k>.pkl. Usage: python seeds.py <fold> [<fold> ...]
"""
import sys, pickle, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'src'))
from postproc import upsample, score_file      # noqa: E402
from common import PQ                          # noqa: E402
from fastsweep import components               # noqa: E402

OUT = Path(__file__).resolve().parent
MIN_SEED = 100


def prob_paths(k):
    a = REPO / ('runs/v2/probs_f%d.npz' % k if k < 2 else 'runs/v2b/probs_f%d.npz' % k)
    b = REPO / ('runs/v2_5070ti/probs_f%d.npz' % k if k < 2 else 'runs/v2b_5070ti/probs_f%d.npz' % k)
    return a, b


def load_fold(k):
    a, b = prob_paths(k)
    za, zb = np.load(a), np.load(b)
    fa, fb = [str(x) for x in za['val_files']], [str(x) for x in zb['val_files']]
    assert fa == fb
    return fa, za['val_prob'], zb['val_prob']


def avg_map(pk_u8, pc_u8):
    """float average (0..255 scale) of the two upsampled maps; also returns the two upsampled maps."""
    uk, uc = upsample(pk_u8), upsample(pc_u8)
    return (uk.astype(np.float32) + uc.astype(np.float32)) * 0.5, uk, uc


def seeds_of(prob):
    """label map of seeds with >= MIN_SEED px, relabelled 1..n, plus (area, mean, peak) per seed."""
    lab, feats = components(prob, 0.4, 8, bridge=0.1)
    big = np.flatnonzero(feats[:, 0] >= MIN_SEED)
    remap = np.zeros(len(feats) + 1, np.int32)
    remap[big + 1] = np.arange(1, len(big) + 1)
    return remap[lab], feats[big], len(feats)


def iou_tables(lab, recs):
    """per record: (n_gt, n_seed) IoU, same arithmetic as postproc.score_file."""
    flat = lab.ravel()
    k = int(flat.max())
    areas = np.bincount(flat, minlength=k + 1)
    out = []
    for rec in recs:
        iou = np.zeros((len(rec), k), np.float32)
        for i, ix in enumerate(rec):
            inter = np.bincount(flat[ix], minlength=k + 1)
            union = areas + len(ix) - inter
            iou[i] = (inter / np.maximum(union, 1))[1:]
            iou[i][inter[1:] == 0] = 0
        out.append(iou)
    return out


def keep_labels(lab, keep):
    """label map holding only the kept seeds (keep: bool per seed), relabelled 1..m."""
    remap = np.zeros(len(keep) + 1, np.int32)
    remap[1:][keep] = np.arange(1, int(keep.sum()) + 1)
    return remap[lab]


def main(folds):
    gt = pickle.load(open(REPO / 'scratch/bank/gt_index.pkl', 'rb'))
    for k in folds:
        t0 = time.time()
        files, PK, PC = load_fold(k)
        rows = []
        tot = PQ()
        for i, f in enumerate(files):
            prob, uk, uc = avg_map(PK[i], PC[i])
            lab, feats, n_all = seeds_of(prob)
            recs = [r[1] for r in gt[f]]
            ious = iou_tables(lab, recs)
            keep = (feats[:, 0] >= 300) & (feats[:, 1] >= 0.85)
            pq = score_file(keep_labels(lab, keep), recs, PQ())
            tot.merge(pq)
            flat = lab.ravel()
            pix = np.flatnonzero(flat).astype(np.int32)
            order = np.argsort(flat[pix], kind='stable')
            pix = pix[order]
            cnt = np.bincount(flat[pix], minlength=len(feats) + 1)[1:]
            assert np.array_equal(cnt, feats[:, 0].astype(np.int64))
            rows.append(dict(fname=f, fold=k, n_rec=len(recs), rec_ids=[r[0] for r in gt[f]],
                             n_gt=[len(r) for r in recs], n_comp_all=n_all,
                             pix=pix, cnt=cnt.astype(np.int32), area=feats[:, 0].copy(), mean=feats[:, 1].copy(),
                             peak=feats[:, 2].copy(), iou=ious, fixed=(pq.tp, pq.fp, pq.fn, pq.s)))
            if i % 20 == 0:
                print('fold', k, i, len(files), f, 'seeds', len(feats), 'running', tot, round(time.time() - t0), 's', flush=True)
        pickle.dump(rows, open(OUT / ('seeds_f%d.pkl' % k), 'wb'), protocol=4)
        print('FOLD', k, 'fixed rule', tot, 'seeds>=100:', sum(len(r['area']) for r in rows), round(time.time() - t0), 's', flush=True)


if __name__ == '__main__':
    main([int(a) for a in sys.argv[1:]])
