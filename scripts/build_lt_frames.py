"""Pack the downloaded neighbour frames (experiment LT) into one array at the training size.

frames_<size>.npy: uint8 (K, size, size), same resize as train_seg.read_gray; index.json: training file name -> rows.
"""
import json, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[1]
out = ROOT / 'data/lt_frames'; size = int(sys.argv[1]) if len(sys.argv) > 1 else 1536
marks = sorted((out / 'done').glob('*.json'))
items = [(json.loads(m.read_text())) for m in marks]
rows = [(it['file'], fr['frame']) for it in items for fr in it['frames'] if (out / 'raw' / f"{fr['frame']}.jpg").exists()]
arr = np.lib.format.open_memmap(out / f'frames_{size}.npy', mode='w+', dtype=np.uint8, shape=(len(rows), size, size))
index = {}
for k, (f, frame) in enumerate(rows):
    im = cv2.imread(str(out / 'raw' / f'{frame}.jpg'), cv2.IMREAD_GRAYSCALE)
    assert im.shape == (2048, 2048), frame
    arr[k] = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    index.setdefault(f, []).append(k)
arr.flush()
(out / 'index.json').write_text(json.dumps(index))
print(json.dumps(dict(frames=len(rows), images_with_frames=len(index), size=size)))
