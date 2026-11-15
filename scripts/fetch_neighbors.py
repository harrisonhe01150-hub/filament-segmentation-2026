"""Label transfer (experiment LT): download GONG archive frames taken by the SAME site within a few minutes of each
labelled TRAINING image, so the provided training annotations can be reused on them (host ruling: discussion 748045).
Only images are downloaded. Any training image within 70 minutes of a test image timestamp is skipped entirely, so no
downloaded frame lies within +/- 1 hour of a test observation. Only explicit file URLs are requested (no crawling).

  python scripts/fetch_neighbors.py --out data/lt_frames --per 4
"""
import argparse, datetime as dt, json, time, urllib.request, urllib.error
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/MAGFiLO_1.0_Kaggle_2026'
BASE = 'https://gong2.nso.edu/HA/hag'


def stamp(name):
    return dt.datetime.strptime(name[:14], '%Y%m%d%H%M%S')


def fetch(url):
    """bytes, or None for 404; raises on throttling so the caller can back off"""
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=ROOT / 'data/lt_frames')
    p.add_argument('--per', type=int, default=4)
    p.add_argument('--shard', type=int, default=0); p.add_argument('--nshard', type=int, default=1)
    a = p.parse_args()
    raw = a.out / 'raw'; raw.mkdir(parents=True, exist_ok=True)
    train = sorted(x.name for x in (DATA / 'train/train_images').glob('*.jpeg'))
    test_t = sorted(stamp(x.name) for x in (DATA / 'test/test_images').glob('*.jpeg'))
    log = open(a.out / f'fetch_{a.shard}.log', 'a')
    offsets = [-2, 2, -4, 4, -3, 3, -5, 5, -1, 1, -6, 6]
    done = skipped = got = 0; t0 = time.time(); backoff = 0
    for f in train[a.shard::a.nshard]:
        t = stamp(f); site = f[14]
        if any(abs((t - x).total_seconds()) <= 70 * 60 for x in test_t):
            skipped += 1
            print(json.dumps(dict(file=f, skipped='within 70 min of a test image')), file=log, flush=True); continue
        mark = a.out / 'done' / (f[:15] + '.json'); mark.parent.mkdir(exist_ok=True)
        if mark.exists():
            done += 1; continue
        ref = cv2.imread(str(DATA / 'train/train_images' / f), cv2.IMREAD_GRAYSCALE)[512:1536, 512:1536].astype(np.float32)
        kept = []
        for off in offsets:
            if len(kept) >= a.per:
                break
            n = (t + dt.timedelta(minutes=off)).strftime('%Y%m%d%H%M%S') + site + 'h'
            url = f'{BASE}/{n[:6]}/{n[:8]}/{n}.jpg'
            try:
                b = fetch(url); backoff = 0
            except Exception as e:  # throttled or network trouble: back off, then give this frame up
                backoff = min(300, 30 * (backoff // 30 + 1))
                print(json.dumps(dict(file=f, url=url, error=str(e)[:120], sleep=backoff)), file=log, flush=True)
                time.sleep(backoff); continue
            if b is None:
                continue
            im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE)
            if im is None or im.shape != (2048, 2048):
                continue
            c = im[512:1536, 512:1536].astype(np.float32)
            corr = float(np.corrcoef(ref.ravel(), c.ravel())[0, 1])
            (dx, dy), _ = cv2.phaseCorrelate(ref - ref.mean(), c - c.mean(), cv2.createHanningWindow((1024, 1024), cv2.CV_32F))
            ok = corr >= .6 and abs(dx) <= 2 and abs(dy) <= 2
            if ok:
                (raw / f'{n}.jpg').write_bytes(b); kept.append(dict(frame=n, minutes=off, corr=round(corr, 4), dx=round(dx, 2), dy=round(dy, 2)))
            else:
                print(json.dumps(dict(file=f, rejected=n, corr=round(corr, 4), dx=round(dx, 2), dy=round(dy, 2))), file=log, flush=True)
            time.sleep(.15)
        mark.write_text(json.dumps(dict(file=f, frames=kept)))
        done += 1; got += len(kept)
        if done % 20 == 0:
            print(json.dumps(dict(shard=a.shard, done=done, frames=got, skipped=skipped, secs=round(time.time() - t0))), flush=True)
    print(json.dumps(dict(shard=a.shard, finished=True, done=done, frames=got, skipped=skipped, secs=round(time.time() - t0))), flush=True)


if __name__ == '__main__':
    main()
