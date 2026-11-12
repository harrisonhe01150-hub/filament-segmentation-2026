"""The candidate keep model of explorer "stronger keep model" as one small class (what src/keep.fit would become).

    score = mean of four predictions of u (u = mean over the image's records of IoU if IoU > 0.5 else 0):
      1. boosting (keep.GBM) on K1's 50 descriptors            -> exactly the present K1 model
      2. boosting (keep.GBM) on the 66 descriptors of scratch/ws/k1_reimpl/feats.py (image_feats)
      3. mean of 5 small networks on the rank-normal 50 descriptors
      4. mean of 5 small networks on the rank-normal 66 descriptors
    all trained with sample weight = number of annotator records of the image; a seed is kept when score > tau.

Both descriptor sets use only the image, the two dense maps and the seeds of the same image.

  python candidate.py --check      refits fold 0 of the original folds and compares with the cached cross-fit
  python candidate.py --table blend_table.npz   writes the training table of the final model (all 6788 seeds)
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import QuantileTransformer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / 'src'))
import keep

MLP = dict(hidden_layer_sizes=(64, 32), alpha=1.0, learning_rate_init=2e-3, max_iter=300, early_stopping=False)
N_NETS = 5


class RankNets:
    """mean of N_NETS networks on the normal scores of the descriptors' ranks among the training rows"""

    def fit(self, X, u, R, seeds):
        self.q = QuantileTransformer(n_quantiles=min(1000, len(X)), output_distribution='normal', random_state=0).fit(X)
        a = np.clip(self.q.transform(X), -4, 4)
        self.nets = [MLPRegressor(random_state=int(s), **MLP).fit(a, u, sample_weight=np.asarray(R, np.float64)) for s in seeds]
        return self

    def predict(self, X):
        a = np.clip(self.q.transform(X), -4, 4)
        return np.mean([m.predict(a) for m in self.nets], 0)


class Blend:
    def fit(self, X50, X66, u, R, seeds50=range(N_NETS), seeds66=range(100, 100 + N_NETS)):
        w = np.asarray(R, np.float64)
        self.g50 = HistGradientBoostingRegressor(**keep.GBM).fit(X50, u, sample_weight=w)
        self.g66 = HistGradientBoostingRegressor(**keep.GBM).fit(X66, u, sample_weight=w)
        self.n50 = RankNets().fit(X50, u, w, seeds50)
        self.n66 = RankNets().fit(X66, u, w, seeds66)
        return self

    def members(self, X50, X66):
        return np.stack([self.g50.predict(X50), self.g66.predict(X66), self.n50.predict(X50), self.n66.predict(X66)])

    def predict(self, X50, X66):
        return self.members(X50, X66).mean(0)


def main():
    import argparse
    import warnings
    warnings.filterwarnings('ignore')
    import xk
    import learners
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--table', type=Path)
    a = ap.parse_args()
    D = xk.Data()
    X66 = learners.load_r66(D)['R66']
    if a.check:
        tr, te = np.flatnonzero(D.fold0 != 0), np.flatnonzero(D.fold0 == 0)
        s50 = [learners.seed_of(tr, 'mlp', s) % 100000 for s in range(N_NETS)]
        s66 = [learners.seed_of(tr, 'mlp_r66', s) % 100000 for s in range(N_NETS)]
        got = Blend().fit(D.X[tr], X66[tr], D.u[tr], D.R[tr], s50, s66).predict(D.X[te], X66[te])
        want = np.mean([np.load(xk.CACHE / f'{n}__orig.npz')['outer'][te] for n in ('k1', 'r66', 'mlp', 'mlp_r66')], 0)
        print('fold 0 of the original folds: max abs difference to the cached cross-fit', float(np.abs(got - want).max()), ' seeds', len(te))
    if a.table:
        with np.load(HERE / 'reimpl66.npz') as z:
            names66 = z['names']
        np.savez_compressed(a.table, X50=D.X.astype(np.float32), X66=X66.astype(np.float32), u=D.u.astype(np.float32), R=D.R.astype(np.int8),
                            names50=np.array(keep.NAMES), names66=names66, fname=D.fname, label=D.label)
        print('wrote', a.table, D.X.shape, X66.shape)


if __name__ == '__main__':
    main()
