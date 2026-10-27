"""Download unlabeled GONG H-alpha frames for self-training (experiment S1, EXPERIMENTS.md).

One frame per day from the public GONG archive (gong2.nso.edu/HA/hag, the same product as the competition
images), years drawn like the training set, skipping every day within 3 days of a test image and within
1 day of a training image. Frames whose disk statistics fall outside the training set's 1-99% range
(clouds, bad frames) are rejected. Only images are downloaded; no annotations exist or are used.

  python scripts/fetch_gong.py --n 700 --out data/gong_unlabeled
"""
import argparse
import datetime as dt
import json
import random
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/MAGFiLO_1.0_Kaggle_2026'
BASE = 'https://gong2.nso.edu/HA/hag'


def day(name):
    return dt.date(int(name[:4]), int(name[4:6]), int(name[6:8]))


def near(days, radius):
    return {d + dt.timedelta(k) for d in days for k in range(-radius, radius + 1)}


def stats(gray):
    c = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)[128:384, 128:384].astype(np.float32)
    return float(c.mean()), float(c.std())


def get(url, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            time.sleep(2 * (i + 1))
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=700)
    p.add_argument('--out', type=Path, default=ROOT / 'data/gong_unlabeled')
    p.add_argument('--seed', type=int, default=2026)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    train = sorted(x.name for x in (DATA / 'train/train_images').glob('*.jpeg'))
    test = sorted(x.name for x in (DATA / 'test/test_images').glob('*.jpeg'))
    banned = near({day(f) for f in test}, 3) | near({day(f) for f in train}, 1)
    ref = np.array([stats(cv2.imread(str(DATA / 'train/train_images' / f), cv2.IMREAD_GRAYSCALE)) for f in train])
    lo, hi = np.percentile(ref, 1, axis=0), np.percentile(ref, 99, axis=0)
    years = Counter(f[:4] for f in train)
    rng = random.Random(a.seed)
    candidates = {}
    for y in years:
        d0 = dt.date(int(y), 1, 1)
        candidates[y] = [d0 + dt.timedelta(k) for k in range((dt.date(int(y), 12, 31) - d0).days + 1)
                         if d0 + dt.timedelta(k) not in banned]
        rng.shuffle(candidates[y])
    want = {y: max(1, round(a.n * c / len(train))) for y, c in years.items()}
    log = dict(accepted=[], rejected=[], want_per_year=want, stats_range=dict(lo=lo.tolist(), hi=hi.tolist()))
    for y in sorted(want):
        got = 0
        for d in candidates[y]:
            if got >= want[y]:
                break
            listing = get(f'{BASE}/{d:%Y%m}/{d:%Y%m%d}/')
            time.sleep(.3)
            if not listing:
                continue
            names = sorted(set(re.findall(rb'(\d{14}[A-Z]h\.jpg)', listing)))
            if not names:
                continue
            name = rng.choice(names).decode()
            raw = get(f'{BASE}/{d:%Y%m}/{d:%Y%m%d}/{name}')
            time.sleep(.3)
            if not raw:
                continue
            gray = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
            if gray is None or gray.shape != (2048, 2048):
                log['rejected'].append(dict(name=name, reason='shape')); continue
            s = np.array(stats(gray))
            if np.any(s < lo) or np.any(s > hi):
                log['rejected'].append(dict(name=name, reason='stats', mean=round(s[0], 1), std=round(s[1], 1))); continue
            cv2.imwrite(str(a.out / name.replace('.jpg', '.png')), gray)  # lossless re-save, grayscale
            log['accepted'].append(name); got += 1
        (a.out / 'MANIFEST.json').write_text(json.dumps(log, indent=1))
        print(json.dumps(dict(year=y, wanted=want[y], got=got, total=len(log['accepted']), rejected=len(log['rejected']))), flush=True)
    clash = {day(n) for n in log['accepted']} & banned
    assert not clash, f'banned days downloaded: {sorted(clash)[:5]}'
    print(json.dumps(dict(done=len(log['accepted']), rejected=len(log['rejected']))), flush=True)


if __name__ == '__main__':
    main()
