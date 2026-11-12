"""Fast post-processing search: compute components once per (t_lo, t_hi seed, close), then filter.

Removing a component never changes other components' IoU, so any filter on component
features (area, mean prob, peak prob) is evaluated exactly from cached IoU matrices.
"""
import sys, json, pickle, itertools, time
from multiprocessing import Pool, get_start_method
import numpy as np
import cv2
from scipy import ndimage as ndi
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from postproc import gt_index, upsample
from common import PQ

GT = None
PROBS = None
FILES = None


def load(prob_paths):
    """Average runs (each a list of fold npz files) at the largest native resolution."""
    runs = []
    for paths in prob_paths:
        p, fl = [], []
        for path in paths:
            z = np.load(path)
            p.append(z['val_prob']); fl += list(z['val_files'])
        runs.append((np.concatenate(p), fl))
    files = runs[0][1]
    S = max(r[0].shape[1] for r in runs)
    for _, fl in runs:
        assert fl == files
    probs = np.zeros((len(files), S, S), np.uint8)
    for i in range(len(files)):  # per-image float accumulation keeps memory bounded
        acc = np.zeros((S, S), np.float32)
        for p, _ in runs:
            acc += p[i] if p.shape[1] == S else cv2.resize(p[i], (S, S), interpolation=cv2.INTER_LINEAR)
        probs[i] = np.round(acc / len(runs)).astype(np.uint8)
    return probs, files


def components(prob, t_lo, close, bridge=0.0):
    """Candidate instances of a probability map (0-255 scale): pixels >= t_lo, grouped into filaments.

    Two pieces belong to one filament if a closing of radius `close` joins them, or (bridge > 0) if they are
    connected through fainter pixels with probability >= bridge. Annotators draw one outline over a filament
    whose dark pieces are linked by faint material; the bridge keeps such pieces together without making the
    mask itself any larger. Returns the label map (only pixels >= t_lo carry a label) and per-label
    (area, mean probability, peak probability)."""
    lo = prob >= t_lo * 255
    grp = lo
    if close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close + 1, 2 * close + 1))
        grp = cv2.morphologyEx(lo.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    if bridge > 0:
        grp = grp | (prob >= bridge * 255)
    lab, n = ndi.label(grp, np.ones((3, 3)))
    lab = (lab * lo).astype(np.int32)
    idx = np.arange(1, n + 1)
    if n == 0:
        return lab, np.zeros((0, 3), np.float32)
    area = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    mean = ndi.mean(prob, lab, idx) / 255
    peak = ndi.maximum(prob, lab, idx) / 255
    feats = np.stack([area, np.nan_to_num(mean), np.nan_to_num(peak)], 1).astype(np.float32)
    return lab, feats


def image_table(args):
    i, t_lo, close = args
    prob = upsample(PROBS[i])
    lab, feats = components(prob, t_lo, close)
    flat = lab.ravel(); k = len(feats)
    areas = np.bincount(flat, minlength=k + 1)
    recs = []
    for rec in GT[FILES[i]]:
        iou = np.zeros((len(rec), k), np.float32)
        for g, ix in enumerate(rec):
            inter = np.bincount(flat[ix], minlength=k + 1)
            union = areas + len(ix) - inter
            iou[g] = np.where(inter[1:] > 0, inter[1:] / np.maximum(union[1:], 1), 0)
        recs.append(iou)
    return feats, recs


def build(t_lo, close):
    tasks = [(i, t_lo, close) for i in range(len(FILES))]
    # Windows/spawn cannot inherit these large globals; serial avoids copying gigabytes.
    if get_start_method() != 'fork':
        return [image_table(task) for task in tasks]
    with Pool(2) as pool:
        return pool.map(image_table, tasks, chunksize=4)


def _init(p, f):
    global PROBS, FILES, GT
    PROBS, FILES = p, f
    GT = gt_index()


def score(tables, min_area=0, min_mean=0, min_peak=0):
    pq = PQ()
    for feats, recs in tables:
        keep = (feats[:, 0] >= min_area) & (feats[:, 1] >= min_mean) & (feats[:, 2] >= min_peak)
        for iou in recs:
            pq.add(iou[:, keep])
    return pq


if __name__ == '__main__':
    spec = json.loads(sys.argv[1])
    PROBS, FILES = load(spec['runs'])
    GT = gt_index()
    results = []
    for t_lo, close in itertools.product(spec['t_lo'], spec['close']):
        t = time.time()
        tables = build(t_lo, close)
        for ma, mm, mp in itertools.product(spec['min_area'], spec['min_mean'], spec['min_peak']):
            pq = score(tables, ma, mm, mp)
            results.append((dict(t_lo=t_lo, close=close, min_area=ma, min_mean=mm, min_peak=mp), pq))
        best = max((r for r in results if r[0]['t_lo'] == t_lo and r[0]['close'] == close), key=lambda r: r[1].value)
        print('built', t_lo, close, round(time.time() - t), 'best', json.dumps(best[0]), best[1], flush=True)
    results.sort(key=lambda r: -r[1].value)
    print('TOP')
    for c, pq in results[:12]:
        print(json.dumps(c), pq)
    out = spec.get('save')
    if out:
        pickle.dump([(c, pq.value, pq.tp, pq.fp, pq.fn) for c, pq in results], open(out, 'wb'))
