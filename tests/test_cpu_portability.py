"""Small synthetic tests for CPU evaluation and cache invalidation."""
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import fastsweep as fs
import postproc as pp

(ROOT / 'scratch').mkdir(exist_ok=True)
tmp = Path(tempfile.mkdtemp(prefix='cpu_', dir=ROOT / 'scratch'))

# Native scipy/OpenCV path, one synthetic image and one expert record.
p = np.zeros((1, 16, 16), np.uint8)
p[0, 4:8, 4:8] = 255
fs.PROBS, fs.FILES = p, ['synthetic.jpeg']
fs.GT = {'synthetic.jpeg': [[np.array([0, 1, 2], np.int32)]]}
direct = fs.image_table((0, .4, 0))
via_build = fs.build(.4, 0)[0]
assert np.array_equal(direct[0], via_build[0])
assert all(np.array_equal(a, b) for a, b in zip(direct[1], via_build[1]))

# The same cwd/cache path must not reuse labels from changed annotations.
pp.ANN = tmp / 'annotations.json'
pp.GT_CACHE = tmp / 'gt_cache.pkl'
calls = []
def load():
    calls.append(pp.ANN.read_text())
    name = 'first' if len(calls) == 1 else 'second'
    return [], defaultdict(list), {name: [1]}
pp.load_coco = load
pp.ANN.write_text('{"version":1}')
assert set(pp.gt_index()) == {'first'}
assert set(pp.gt_index()) == {'first'} and len(calls) == 1
pp.ANN.write_text('{"version":2}')
assert set(pp.gt_index()) == {'second'} and len(calls) == 2
assert len(list(tmp.glob('gt_cache_*_v1.pkl'))) == 2
result = {'windows_cpu_path_matches_direct': True, 'cache_reused_for_same_input': True,
          'changed_annotation_invalidates_cache': True, 'original_cache_preserved': True,
          'real_annotations_read': False}
(ROOT.parent / 'CPU_TEST_RESULTS.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
