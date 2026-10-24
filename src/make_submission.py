"""Average test probability maps across fold models/runs, extract instances, write submission CSV."""
import sys, json
import numpy as np
import cv2
from pycocotools import mask as mu
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from fastsweep import components


def main(spec):
    stacks, files = [], None
    for path in spec['npz']:
        z = np.load(path)
        fl = list(z['test_files'])
        assert files is None or fl == files
        files = fl
        stacks.append(z['test_prob'])  # uint8, native model resolution
    c = spec['cfg']
    rows, empty = [], []
    for i, f in enumerate(files):
        # average at 2048 per image (memory-safe); same upsampling as OOF evaluation
        prob = np.mean([cv2.resize(s[i], (2048, 2048), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                        for s in stacks], 0)
        lab, feats = components(prob, c['t_lo'], c['close'])
        keep = (feats[:, 0] >= c['min_area']) & (feats[:, 1] >= c['min_mean']) & (feats[:, 2] >= c['min_peak'])
        stem = f.rsplit('.', 1)[0]
        n = 0
        for j in np.flatnonzero(keep):
            m = np.asfortranarray((lab == j + 1).astype(np.uint8))
            rle = mu.encode(m)
            n += 1
            rows.append((f'{stem}_{n}', rle['counts'].decode()))
        if n == 0:
            empty.append(f)
    with open(spec['out'], 'w') as fh:
        fh.write('filament_id,segmentation_rle\n')
        for a, b in rows:
            fh.write(f'{a},{b}\n')
    print(json.dumps(dict(rows=len(rows), images=len(files), empty_images=empty)))


if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
