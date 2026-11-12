"""Paste refined crops back into the images and score the result out of fold. CPU only.

Step 1, per fold (independent processes; without --fold all five run one after another):
  python analysis/refine_eval.py --bank scratch/ws/refine/bank/oof --out scratch/ws/refine/eval --source stage1 --fold 0
Step 2, the tables:
  python analysis/refine_eval.py --out scratch/ws/refine/eval --source refined --report
Step 3, a submission from a TEST bank (analysis/refine_bank.py with "test") and the refined crops of its seeds
(analysis/refine_train.py --apply), with the paste setting chosen in step 2:
  python analysis/refine_eval.py --bank scratch/ws/refine/bank/test --pred scratch/ws/refine/sub/pred_pair
      --submit outputs/refined.csv --paste '{"t_r": 0.6, "margin": 0.3, "t_join": 0.5, "absorb": 0.3}'
Without --pred the stage-1 seeds are written, which must equal the fixed-rule submission mask for mask.

--source says what replaces each kept stage-1 seed:
  stage1     nothing: the seeds themselves. Must reproduce the fixed-rule baseline exactly (sanity test A).
  roundtrip  the seed's own crop pasted back. Measures what cutting and pasting alone costs (sanity test B).
  oracle     the seed's training target pasted back. Upper bound of a perfect refiner (sanity test C).
  refined    the refiner's prediction (--pred = folder with pred_f<fold>.npz from analysis/refine_train.py),
             for every paste setting in `settings_of`.

Two scoring rules for every label map:
  official   organisers' self-evaluation notebook: each annotator record scored separately, counts pooled.
             This is the rule of the leaderboard, and every choice below is made with it.
  pooled     all records of an image form one ground-truth set; a prediction matches at most its best filament
             (a side number only).
The paste settings are chosen in sample (all folds) and nested (for fold k on the other four folds); the nested
number is the honest one. This is done separately for the families of settings in FAMILIES, so the tables show
what the design alone gives and what the joining rules and the margin add.
"""
import argparse
import json
import pickle
import sys
import time
from collections import namedtuple
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'src'))
sys.path.insert(0, str(WORK / 'analysis'))
from tree_bank import gt_index
import refine
from common import PQ
from make_submission import check_disjoint, write_csv
from postproc import score_file

GT_INDEX = WORK / 'scratch/bank/gt_index.pkl'      # TRAIN annotations as pixel lists; built on first use (about 2 minutes)
SHAPE = refine.IMAGE_SHAPE
KEEP_SHAPE = float('inf')                          # margin that never changes a seed (see refine.paste_image)

# One way of pasting; the fields are the arguments of refine.paste_image.
Setting = namedtuple('Setting', 't_r fill margin t_join cover absorb')
T_R = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
MARGINS = (0.1, 0.2, 0.3, 0.4)
RULES = ((0.3, None), (0.5, None), (None, 0.3), (None, 0.5))     # joining rules as (cover, absorb)
T_JOIN = (0.4, 0.5, 0.6)                                         # plain refined masks that may decide the joins
FAMILIES = (
    ('design', 'threshold and hole filling', lambda s: not (s.margin or s.cover or s.absorb)),
    ('joins', 'the same + joining rules', lambda s: not s.margin),
    ('margin', 'the same + margin', lambda s: s.margin != KEEP_SHAPE),
    ('keep', 'stage-1 shapes, joined by the refined masks', lambda s: s.margin == KEEP_SHAPE),
    ('all', 'all settings', lambda s: True),
)


def settings_of(source):
    """Paste settings evaluated for a source. Only the refiner has free settings."""
    if source != 'refined':
        return [Setting(0.5, False, 0.0, None, None, None)]
    out = [Setting(t, fill, 0.0, None, None, None) for fill in (False, True) for t in T_R]
    out += [Setting(t, False, margin, None, None, None) for margin in MARGINS for t in T_R]
    for margin in (0.0,) + MARGINS:
        out += [Setting(t, False, margin, None, *rule) for rule in RULES for t in T_R]
        out += [Setting(t, False, margin, t_join, *rule) for t_join in T_JOIN for rule in ((0.3, None), (None, 0.3)) for t in T_R]
    out += [Setting(0.5, False, KEEP_SHAPE, t_join, *rule) for t_join in (0.3,) + T_JOIN for rule in RULES + ((None, 0.2),)]
    return out


def name_of(s):
    shape = 'stage-1 shapes' if s.margin == KEEP_SHAPE else f't_r {s.t_r:.2f}' + (f' margin {s.margin}' if s.margin else '')
    return (shape + (' fill' if s.fill else '') + (f' cover {s.cover}' if s.cover else '') + (f' absorb {s.absorb}' if s.absorb else '')
            + (f' joined at {s.t_join}' if s.t_join else ''))


def iou_tables(labels, areas, recs):
    """IoU of every drawn filament with every instance: one (filaments, instances) matrix per annotator record.
    areas: pixel count of each instance (refine.assemble returns it, which saves a pass over the image)."""
    flat = labels.ravel()
    k = len(areas)
    out = []
    for _, filaments in recs:
        iou = np.zeros((len(filaments), k), np.float32)
        for g, ix in enumerate(filaments):
            inter = np.bincount(flat[ix], minlength=k + 1)[1:]
            iou[g] = np.where(inter > 0, inter / np.maximum(areas + len(ix) - inter, 1), 0)
        out.append(iou)
    return out


def pooled_rule(tables):
    """(TP, FP, FN, sum of matched IoU) with all records of the image as one ground-truth set."""
    n_gt = sum(t.shape[0] for t in tables)
    k = tables[0].shape[1]
    if k == 0 or n_gt == 0:
        return np.array([0, k, n_gt, 0.0])
    best = np.concatenate(tables, 0).max(0)
    hit = best > 0.5
    return np.array([hit.sum(), k - hit.sum(), n_gt - hit.sum(), best[hit].sum()], np.float64)


def score(labels, areas, recs, check):
    """(2, 4) array: official and pooled (TP, FP, FN, sum of matched IoU) of one label map.

    Both rules are counted from one set of IoU tables. With `check` the official counts are compared with
    postproc.score_file, the implementation of the organisers' rule used everywhere else in this repository.
    """
    tables = iou_tables(labels, areas, recs)
    pq = PQ()
    for table in tables:
        pq.add(table)
    if check:
        ref = score_file(labels, [filaments for _, filaments in recs], PQ())
        assert (ref.tp, ref.fp, ref.fn) == (pq.tp, pq.fp, pq.fn) and abs(ref.s - pq.s) < 1e-6, 'scoring differs from score_file'
    return np.array([[pq.tp, pq.fp, pq.fn, pq.s], pooled_rule(tables)], np.float64)


def pq_of(t):
    d = t[0] + 0.5 * t[1] + 0.5 * t[2]
    return float(t[3] / d) if d else 0.0


def run_fold(bank_dir, fold, source, pred_dir, out_dir):
    t0 = time.time()
    keys = ['fname', 'label', 'kept', 'M', 'pix', 'seed', 'scale']
    if source == 'oracle':
        keys += ['target', 'n_match']
    bank = refine.load_bank(bank_dir, keys, fold)
    images = refine.load_images(bank_dir, fold)
    crops = None
    if source == 'roundtrip':
        crops = bank['seed']
    elif source == 'oracle':
        crops = bank['target']
    elif source == 'refined':
        with np.load(Path(pred_dir) / f'pred_f{fold}.npz', allow_pickle=False) as z:
            if not (np.array_equal(z['fname'], bank['fname']) and np.array_equal(z['label'], bank['label'])):
                raise ValueError('predictions and bank list different seeds')
            crops = z['pred']
    gt = gt_index(GT_INDEX)
    start = {f: i for i, f in reversed(list(enumerate(bank['fname'])))}
    count = {f: int(n) for f, n in zip(*np.unique(bank['fname'], return_counts=True))}
    settings = settings_of(source)
    stats = np.zeros((len(images), len(settings), 2, 4))
    n_inst = np.zeros((len(images), len(settings)), np.int32)
    n_unchanged = np.zeros((len(images), len(settings)), np.int32)
    n_kept = np.zeros(len(images), np.int32)
    per_seed = []       # roundtrip only: (scale, IoU of the pasted mask with the stage-1 mask)
    for n, f in enumerate(images):
        rows = [i for i in range(start.get(f, 0), start.get(f, 0) + count.get(f, 0)) if bank['kept'][i]]
        n_kept[n] = len(rows)
        pix = [bank['pix'][i] for i in rows]
        refined = [crops[i] if crops is not None and (source != 'oracle' or bank['n_match'][i] > 0) else None for i in rows]
        cache = {}      # (threshold, fill holes, margin) -> pieces of all seeds; shared by the joining rules

        def pieces(t, fill, margin):
            if margin == KEEP_SHAPE:
                return [None] * len(rows)
            if (t, fill, margin) not in cache:
                cache[t, fill, margin] = [None if r is None else refine.refined_piece(r, bank['seed'][i], bank['M'][i], t, SHAPE, fill, margin)
                                          for i, r in zip(rows, refined)]
            return cache[t, fill, margin]

        for c, s in enumerate(settings):
            join = None if s.t_join is None else pieces(s.t_join, False, 0.0)
            labels, areas, unchanged = refine.assemble(SHAPE, pix, pieces(s.t_r, s.fill, s.margin), s.cover, s.absorb, join)
            stats[n, c] = score(labels, areas, gt[f], check=c == 0)
            n_inst[n, c] = len(areas)
            n_unchanged[n, c] = unchanged
        if source == 'roundtrip':
            for i, x, piece in zip(rows, pix, pieces(0.5, False, 0.0)):
                inter = 0 if piece is None else np.intersect1d(piece[0], x, assume_unique=True).size
                per_seed.append((float(bank['scale'][i]), inter / (len(x) + (0 if piece is None else len(piece[0])) - inter)))
        if (n + 1) % 40 == 0:
            print(json.dumps(dict(fold=fold, done=n + 1, of=len(images), secs=round(time.time() - t0))), flush=True)
    out = dict(source=source, fold=fold, images=images, n_rec=np.array([len(gt[f]) for f in images]), settings=[tuple(s) for s in settings],
               stats=stats, n_inst=n_inst, n_unchanged=n_unchanged, n_kept=n_kept, per_seed=np.array(per_seed))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{source}_f{fold}.pkl').write_bytes(pickle.dumps(out))
    total = stats.sum(0)
    best = max(range(len(settings)), key=lambda c: pq_of(total[c, 0]))
    print(json.dumps(dict(fold=fold, source=source, images=len(images), best=name_of(settings[best]),
                          official=round(pq_of(total[best, 0]), 4), secs=round(time.time() - t0))), flush=True)


def load_runs(out_dir, source):
    runs = [pickle.loads((out_dir / f'{source}_f{k}.pkl').read_bytes()) for k in range(5)]
    return dict(settings=[Setting(*s) for s in runs[0]['settings']], fold=np.concatenate([np.full(len(r['images']), r['fold']) for r in runs]),
                **{k: np.concatenate([r[k] for r in runs]) for k in ('n_rec', 'stats', 'n_inst', 'n_unchanged', 'n_kept', 'per_seed')})


HEAD = ('| | official PQ | TP / FP / FN | mean IoU | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | 1 record | 2 records | 3 records | pooled PQ |\n'
        '|---|---|---|---|---|---|---|---|---|---|---|---|---|')


def line(name, per_image, fold, n_rec):
    """One table row from per-image (2, 4) counts: official PQ overall, per fold and per number of records."""
    o, p = per_image.sum(0)
    cells = [f'{pq_of(per_image[fold == k].sum(0)[0]):.4f}' for k in range(5)]
    cells += [f'{pq_of(per_image[n_rec == m].sum(0)[0]):.4f}' for m in (1, 2, 3)]
    return (f'| {name} | {pq_of(o):.4f} | {int(o[0])} / {int(o[1])} / {int(o[2])} | {o[3] / max(o[0], 1):.4f} | '
            + ' | '.join(cells) + f' | {pq_of(p):.4f} |')


def report(out_dir, source):
    r = load_runs(out_dir, source)
    base = load_runs(out_dir, 'stage1') if source != 'stage1' else None
    settings, stats, fold, n_rec = r['settings'], r['stats'], r['fold'], r['n_rec']
    by_fold = np.stack([stats[fold == k].sum(0) for k in range(5)])          # (5, settings, 2, 4)
    total = by_fold.sum(0)
    order = np.concatenate([np.flatnonzero(fold == k) for k in range(5)])
    print(f'\n## {source}: {len(fold)} images, {int(r["n_kept"].sum())} kept seeds, {len(settings)} paste settings\n')
    print(HEAD)
    if base is not None:
        print(line('baseline (stage 1)', base['stats'][:, 0], base['fold'], base['n_rec']))
    result = dict(source=source, families={})
    for key, title, member in FAMILIES if len(settings) > 1 else [('only', source, lambda s: True)]:
        ids = [c for c, s in enumerate(settings) if member(s)]
        best = max(ids, key=lambda c: pq_of(total[c, 0]))
        picks = [max(ids, key=lambda c: pq_of((total[c] - by_fold[k, c])[0])) for k in range(5)]
        nested = np.concatenate([stats[fold == k, picks[k]] for k in range(5)])
        print(line(f'{title}: in-sample best ({name_of(settings[best])})', stats[:, best], fold, n_rec))
        if len(settings) > 1:
            print(line(f'{title}: nested', nested, fold[order], n_rec[order]))
        result['families'][key] = dict(
            title=title, settings=len(ids),
            in_sample=dict(setting=name_of(settings[best]), official=pq_of(total[best, 0]), pooled=pq_of(total[best, 1]),
                           unchanged=int(r['n_unchanged'][:, best].sum()), instances=int(r['n_inst'][:, best].sum())),
            nested=dict(settings=[name_of(settings[c]) for c in picks], official=pq_of(nested.sum(0)[0]), pooled=pq_of(nested.sum(0)[1]),
                        per_fold=[pq_of(nested[fold[order] == k].sum(0)[0]) for k in range(5)],
                        three_records=pq_of(nested[n_rec[order] == 3].sum(0)[0])))
    if len(settings) > 1:
        for f in result['families'].values():
            print(f'\n{f["title"]} ({f["settings"]} settings). In-sample best: {f["in_sample"]["setting"]}; of the kept seeds '
                  f'{f["in_sample"]["unchanged"]} are pasted unchanged and {int(r["n_kept"].sum()) - f["in_sample"]["instances"]} '
                  f'are joined to another instance or left without a pixel. Nested picks per fold: {f["nested"]["settings"]}.')
        print('\nBest 12 settings in sample:\n\n| setting | official PQ | TP / FP / FN | mean IoU | 3 records | pooled PQ |\n|---|---|---|---|---|---|')
        for c in sorted(range(len(settings)), key=lambda c: -pq_of(total[c, 0]))[:12]:
            o = total[c, 0]
            print(f'| {name_of(settings[c])} | {pq_of(o):.4f} | {int(o[0])} / {int(o[1])} / {int(o[2])} | {o[3] / max(o[0], 1):.4f} | '
                  f'{pq_of(stats[n_rec == 3, c].sum(0)[0]):.4f} | {pq_of(total[c, 1]):.4f} |')
    if source == 'roundtrip':
        s, iou = r['per_seed'].T
        print(f'\nRound trip of {len(s)} kept seeds: scale s < 0.5 for {int((s < 0.5).sum())}, 0.5 <= s < 1 for '
              f'{int(((s >= 0.5) & (s < 1)).sum())}, s >= 1 for {int((s >= 1).sum())} (s = 2, the cap, for {int((s >= 2).sum())}).')
        print('| seeds | n | IoU min | 1 % | 10 % | median | mean |\n|---|---|---|---|---|---|---|')
        for label, m in (('s < 0.5', s < 0.5), ('0.5 <= s < 1', (s >= 0.5) & (s < 1)), ('s >= 1', s >= 1), ('all', s > 0)):
            if m.any():
                q = np.percentile(iou[m], [0, 1, 10, 50])
                print(f'| {label} | {int(m.sum())} | {q[0]:.4f} | {q[1]:.4f} | {q[2]:.4f} | {q[3]:.4f} | {iou[m].mean():.4f} |')
    result['all_settings'] = [dict(setting=name_of(s), official=pq_of(total[c, 0]), pooled=pq_of(total[c, 1]),
                                   per_fold=[pq_of(by_fold[k, c, 0]) for k in range(5)],
                                   three_records=pq_of(stats[n_rec == 3, c].sum(0)[0])) for c, s in enumerate(settings)]
    (out_dir / f'report_{source}.json').write_text(json.dumps(result, indent=1))


def submit(bank_dir, pred_dir, paste, out):
    """Write the submission CSV of a test bank and check that no two masks of an image share a pixel."""
    if out.exists():
        raise FileExistsError(f'{out} exists; choose a new path')
    rows, empty = refine.submission_rows(bank_dir, pred_dir, **paste)
    checked = check_disjoint(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out)
    print(json.dumps(dict(rows=len(rows), images_checked_disjoint=checked, empty_images=empty, paste=paste, out=str(out))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', type=Path, default=WORK / 'scratch/ws/refine/bank/oof')
    ap.add_argument('--out', type=Path, help='folder of the per-fold results')
    ap.add_argument('--source', choices=['stage1', 'roundtrip', 'oracle', 'refined'])
    ap.add_argument('--submit', type=Path, help='write a submission CSV for the bank instead of scoring it')
    ap.add_argument('--paste', default='{"t_r": 0.5}', help='JSON arguments of refine.paste_image for --submit')
    ap.add_argument('--pred', type=Path, help='folder with pred_f<fold>.npz (source refined)')
    ap.add_argument('--fold', type=int, help='only this fold (default: all five)')
    ap.add_argument('--report', action='store_true', help='print the tables from the per-fold results in --out')
    a = ap.parse_args()
    if a.submit:
        submit(a.bank, a.pred, json.loads(a.paste), a.submit)
        return
    if not (a.out and a.source):
        ap.error('--out and --source are required unless --submit is given')
    if a.report:
        report(a.out, a.source)
        return
    for fold in range(5) if a.fold is None else [a.fold]:
        run_fold(a.bank, fold, a.source, a.pred, a.out)


if __name__ == '__main__':
    main()
