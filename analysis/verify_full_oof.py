"""Stream the fixed v2 five-fold OOF evaluation, without test prediction access."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'src'))
from common import PQ, load_coco, ann_mask
from postproc import upsample, score_file
from fastsweep import components
from train_seg import fold_of_files


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    root = a.root
    out = root / 'results/full_v2_oof'
    out.mkdir(parents=True, exist_ok=True)
    annotation = root / 'data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
    images, anns, by_file = load_coco(annotation)
    assigned = fold_of_files(by_file)
    total, fold_rows, rows, seen = PQ(), [], [], set()
    start = time.monotonic()
    for fold in range(5):
        folder = 'v2' if fold < 2 else 'v2b'
        log = json.loads((root / 'runs' / folder / f'log_f{fold}.json').read_text())
        assert len(log['log']) == 50 and log['cfg']['encoder'] == 'efficientnet-b4'
        with np.load(root / 'runs' / folder / f'probs_f{fold}.npz', allow_pickle=False) as z:
            files, prob = [str(x) for x in z['val_files']], z['val_prob']
        assert len(files) == len(set(files)) and not set(files).intersection(seen)
        assert set(files) == {f for f, k in assigned.items() if k == fold}
        one = PQ()
        for image, prediction in zip(files, prob):
            labels, feat = components(upsample(prediction), .4, 8)
            keep = (feat[:, 0] >= 300) & (feat[:, 1] >= .85)
            remap = np.zeros(len(feat) + 1, np.int32)
            remap[1:][keep] = np.arange(1, int(keep.sum())+1)
            recs = [[np.flatnonzero(ann_mask(an)).astype(np.int32) for an in anns[r]] for r in by_file[image]]
            scored = score_file(remap[labels], recs, PQ())
            total.merge(scored); one.merge(scored); seen.add(image)
            rows.append(dict(file=image, date=image[:8], fold=fold, pq=scored.value, tp=scored.tp,
                             fp=scored.fp, fn=scored.fn, matched_iou_sum=scored.s, records=len(recs)))
            if len(rows) % 25 == 0:
                print(json.dumps(dict(event='full_oof_progress', images=len(rows), seconds=time.monotonic()-start)), flush=True)
        fold_rows.append(dict(fold=fold, images=len(files), pq=one.value, tp=one.tp, fp=one.fp, fn=one.fn,
                              matched_iou_sum=one.s))
        del prob
    assert seen == set(by_file) and len(seen) == 707
    result = dict(status='COMPLETED', images=707, records=sum(r['records'] for r in rows), folds=fold_rows,
                  pq=total.value, tp=total.tp, fp=total.fp, fn=total.fn, matched_iou_sum=total.s,
                  historical_rounded_reference=.4409, reference_verified=abs(total.value-.4409)<.0001,
                  elapsed_seconds=time.monotonic()-start, gpu_used=False, test_probabilities_used=False,
                  test_labels_used=False, independent_final_validation=False)
    (out/'SUMMARY.json').write_text(json.dumps(result, indent=2))
    (out/'PER_IMAGE.json').write_text(json.dumps(rows, indent=2))
    assert result['reference_verified'], result
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
