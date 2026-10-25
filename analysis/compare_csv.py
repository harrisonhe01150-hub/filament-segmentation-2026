"""Compare two submission CSVs filament by filament (match by IoU > 0.5): python analysis/compare_csv.py a.csv b.csv"""
import csv
import json
import sys
from collections import defaultdict

import numpy as np
from pycocotools import mask as mu


def load(path):
    out = defaultdict(list)
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            out[row['filament_id'].rsplit('_', 1)[0]].append(row['segmentation_rle'])
    return out


a, b = load(sys.argv[1]), load(sys.argv[2])
ious, unmatched_a, unmatched_b, count_diff = [], 0, 0, 0
for image in sorted(set(a) | set(b)):
    ra = [{'size': [2048, 2048], 'counts': c.encode()} for c in a.get(image, [])]
    rb = [{'size': [2048, 2048], 'counts': c.encode()} for c in b.get(image, [])]
    count_diff += len(ra) != len(rb)
    if not ra or not rb:
        unmatched_a += len(ra); unmatched_b += len(rb); continue
    iou = np.array(mu.iou(ra, rb, [0] * len(rb)))
    best = iou.max(1)
    ious += best[best > .5].tolist()
    unmatched_a += int((best <= .5).sum()); unmatched_b += int((iou.max(0) <= .5).sum())
print(json.dumps(dict(rows_a=sum(map(len, a.values())), rows_b=sum(map(len, b.values())), images_with_different_count=count_diff,
                      matched=len(ious), unmatched_a=unmatched_a, unmatched_b=unmatched_b,
                      min_iou=round(min(ious), 5) if ious else None, mean_iou=round(float(np.mean(ious)), 6) if ious else None,
                      share_iou_ge_0999=round(float(np.mean(np.array(ious) >= .999)), 4) if ious else None), indent=2))
