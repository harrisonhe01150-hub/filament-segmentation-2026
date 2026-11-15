# weights/

Files the final pipeline (`src/predict_final.py`, `notebooks/end_to_end_final.ipynb`) reads. All of them are
committed (2026-11-12). Check them after a clone with `sha256sum weights/*` against the table.

| File here | Made by | Copied from (training machine) | Size | SHA256 |
|---|---|---|---|---|
| `keep_table.npz` | `analysis/keep_fit.py --table` (6788 seeds x 50 descriptors from 2 member models, labels from the TRAIN annotations) | committed | 1.0 MB | `e2d35e8a87ff90297d996c4070f836f98e32a16d8c9174014447a4c8adaf030a` |
| `model_full_s2026.pt` | `src/train_seg.py --config configs/v2_full.json` (all 707 images, seed 2026) | `runs/v2_full_5070ti/model_f-1.pt` | 81.7 MB | `ad5647c7f69f2541a5d91140c6d03b956775bb9d3fc5bc9e2ee6af3529bfbdec` |
| `model_full_s2027.pt` | `src/train_seg.py --config configs/v2_full_s2027.json` (seed 2027) | `runs/v2_full_s2027_5070ti/model_f-1.pt` | 81.7 MB | `e616121936b6f6b480c0f2242d89ef554f879734edd17e5617576123b4894bee` |
| `refiner_f0.pt` | `analysis/refine_train.py --fold 0` (seeds of folds 1-4) | `scratch/ws/refine/run1/refiner_f0.pt` | 41.0 MB | `fbfd5d5e40198a20982976ad1723ea56790a11c5f7b71ad26aa3094e8219db96` |
| `refiner_f1.pt` | same, fold 1 | `scratch/ws/refine/run1/refiner_f1.pt` | 41.0 MB | `5c473a471d3b20cf70c3d4fda7fa6d88cd5a3bb2576d0f788ba93b80ad8486d3` |
| `refiner_f2.pt` | same, fold 2 | `scratch/ws/refine/run1/refiner_f2.pt` | 41.0 MB | `e0d47a322d36ca360082f460407c434d81f5d01cb6c8da24640b0d242de7e969` |
| `refiner_f3.pt` | same, fold 3 | `scratch/ws/refine/run1/refiner_f3.pt` | 41.0 MB | `f8a392d311b62e6c9f2ec1f2122e72e06ec1f5f48f11b7c676902e17ebd11e68` |
| `refiner_f4.pt` | same, fold 4 | `scratch/ws/refine/run1/refiner_f4.pt` | 41.0 MB | `2532df7b414f9c9b5ca4af265e9c6e418338bb9e6f100adfcd086b4a4ce0e7ae` |

The lighter variant (`--no-refiner`) needs only the first three rows.

The refiners are float16 state dicts, the dense models float32; every file is under GitHub's 100 MB limit.
