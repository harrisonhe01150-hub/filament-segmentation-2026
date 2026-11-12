"""Build the seed bank for the stage-2 refiner (src/refine.py). CPU only.

For every image: the stage-1 seeds (connected components of the probability map at 0.4) and, for each seed, its
oriented crop of the network inputs. For training images (out-of-fold maps) also the target, i.e. the consensus of
the annotators who drew that filament. Everything later (training, prediction, paste-back) reads this bank, so the
2048 px images and maps are touched only here.

Out-of-fold bank (members = one {fold: run folder} map per stage-1 model; several models are averaged):
  python analysis/refine_bank.py '{"tag": "oof", "members": [{"0": "v2", "1": "v2", "2": "v2b", "3": "v2b", "4": "v2b"}]}'
Test bank (test = probs npz files that hold test_prob; averaged exactly like src/make_submission.py):
  python analysis/refine_bank.py '{"tag": "test", "test": ["runs/v2/probs_f0.npz", "runs/v2/probs_f1.npz"]}'
Optional keys: shard / nshard (independent processes), out (default scratch/ws/refine/bank), folds,
bridge (0.1 = the grouping of predict.POST_STACK: pieces linked by probability >= 0.1 are one seed; default 0).
"""
import json
import pickle
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'src'))
from postproc import upsample
from refine import PLANES, image_bank

DATA = WORK / 'data/MAGFiLO_1.0_Kaggle_2026'
GT_INDEX = WORK / 'scratch/bank/gt_index.pkl'      # written by analysis/tree_bank.py from the TRAIN annotations


def read_maps(path, key, rows):
    """Rows `rows` of the uint8 stack `key` inside an .npz, read as a stream: {row: map}.

    np.load would inflate the whole stack (335 MB per fold) in every shard; the shards run side by side on a
    machine that is also training, so only the wanted maps are kept in memory.
    """
    wanted, out = set(rows), {}
    with zipfile.ZipFile(path) as z, z.open(key + '.npy') as f:
        major, _ = np.lib.format.read_magic(f)
        header = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = header(f)
        if dtype != np.uint8 or fortran:
            raise ValueError(f'{path}:{key} is not a C-ordered uint8 array')
        size = int(np.prod(shape[1:]))
        for i in range(shape[0]):
            buf = f.read(size)
            if i in wanted:
                out[i] = np.frombuffer(buf, np.uint8).reshape(shape[1:])
    return out


def save_bank(path, files, fold, n_recs, items):
    """One .npz of per-seed arrays (`pix` is ragged, so it is stored flat with row pointers) plus `images`, the
    names of all images of this shard, because an image without seeds must still be scored."""
    counts = [len(it['label']) for it in items]
    out = dict(images=np.array(files), fname=np.repeat(np.array(files), counts),
               fold=np.full(sum(counts), fold, np.int8), n_rec=np.repeat(np.array(n_recs, np.int8), counts))
    names = ['label', 'kept', 'area', 'mean', 'M', 'scale', 'long', 'short', *PLANES]
    if fold >= 0:
        names += ['target', 'n_match', 'best_iou']
    for name in names:
        out[name] = np.concatenate([it[name] for it in items])
    pix = [p for it in items for p in it['pix']]
    out['pix'] = np.concatenate(pix) if pix else np.zeros(0, np.int32)
    out['pix_ptr'] = np.concatenate([[0], np.cumsum([len(p) for p in pix])]).astype(np.int64)
    np.savez_compressed(path, **out)


def main():
    spec = json.loads(sys.argv[1])
    out = WORK / spec.get('out', 'scratch/ws/refine/bank') / spec['tag']
    out.mkdir(parents=True, exist_ok=True)
    shard, nshard = spec.get('shard', 0), spec.get('nshard', 1)
    cv2.setNumThreads(1)        # one core per shard; the shards are the parallelism
    t0 = time.time()
    jobs = []  # (fold or -1, [npz paths], key holding the maps, key holding the file names, image folder)
    if 'test' in spec:
        jobs.append((-1, [WORK / p for p in spec['test']], 'test_prob', 'test_files', DATA / 'test/test_images'))
    else:
        for fold in spec.get('folds', range(5)):
            jobs.append((fold, [WORK / 'runs' / m[str(fold)] / f'probs_f{fold}.npz' for m in spec['members']],
                         'val_prob', 'val_files', DATA / 'train/train_images'))
    gt = None if 'test' in spec else pickle.loads(GT_INDEX.read_bytes())
    done = seeds = 0
    for fold, paths, map_key, file_key, folder in jobs:
        target = out / f'bank_f{fold}_s{shard}of{nshard}.npz'
        if target.exists():
            continue
        files = None
        for path in paths:
            with np.load(path, allow_pickle=False) as z:
                names = [str(x) for x in z[file_key]]
            if files is not None and names != files:
                raise ValueError(f'{path} lists other images than {paths[0]}')
            files = names
        rows = range(shard, len(files), nshard)
        maps = [read_maps(path, map_key, rows) for path in paths]
        items = []
        for i in rows:
            # same upsampling and averaging as the fixed-rule pipeline, so the seeds are its components
            members = [upsample(m[i]) for m in maps]
            prob = members[0] if len(members) == 1 else np.mean([m.astype(np.float32) for m in members], 0)
            gray = cv2.imread(str(folder / files[i]), cv2.IMREAD_GRAYSCALE)
            if gray is None or gray.shape != prob.shape:
                raise ValueError(f'cannot read {folder / files[i]} as a {prob.shape} image')
            items.append(image_bank(prob, gray, None if gt is None else gt[files[i]], spec.get('bridge', 0.0)))
            done += 1
            seeds += len(items[-1]['label'])
            if done % 20 == 0:
                print(json.dumps(dict(shard=shard, fold=fold, done=done, seeds=seeds, secs=round(time.time() - t0))), flush=True)
        save_bank(target, [files[i] for i in rows], fold, [0 if gt is None else len(gt[files[i]]) for i in rows], items)
        del maps, items
    print(json.dumps(dict(shard=shard, finished=True, images=done, seeds=seeds, secs=round(time.time() - t0))), flush=True)


if __name__ == '__main__':
    main()
