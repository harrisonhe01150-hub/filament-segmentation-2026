"""Learners of explorer "stronger keep model". learner(D, tr, te) -> predicted u of the seeds te.
Nothing here reads a label of a row in te."""
import zlib

import numpy as np
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestRegressor)

import keep

REG = {}
JOBS = 4


def reg(name):
    def deco(f):
        REG[name] = f
        return f
    return deco


def seed_of(*parts):
    """deterministic 31-bit seed from the training rows and a tag, so every cross-fit cell has its own stream"""
    h = zlib.crc32(np.asarray(parts[0], np.int64).tobytes())
    for p in parts[1:]:
        h = zlib.crc32(str(p).encode(), h)
    return h & 0x7fffffff


def hgb(X, y, w, Xte, **kw):
    p = dict(keep.GBM)
    p.update(kw)
    return HistGradientBoostingRegressor(**p).fit(X, y, sample_weight=w).predict(Xte)


# ------------------------------------------------------------------------------------------------- the baseline

@reg('k1')
def k1(D, tr, te):
    """exactly src/keep.py"""
    return keep.fit(D.X[tr], D.u[tr], D.R[tr]).predict(D.X[te])


# ------------------------------------------------------------------------- other learners, same descriptors and target

@reg('rf')
def rf(D, tr, te):
    m = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, max_features=1 / 3, n_jobs=JOBS, random_state=seed_of(tr, 'rf'))
    return m.fit(D.X[tr], D.u[tr], sample_weight=D.w[tr]).predict(D.X[te])


@reg('et')
def et(D, tr, te):
    m = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.5, n_jobs=JOBS, random_state=seed_of(tr, 'et'))
    return m.fit(D.X[tr], D.u[tr], sample_weight=D.w[tr]).predict(D.X[te])


def rank_normal(Xtr, Xte):
    """descriptors -> normal scores of their rank among the training rows (no labels involved)"""
    from sklearn.preprocessing import QuantileTransformer
    q = QuantileTransformer(n_quantiles=min(1000, len(Xtr)), output_distribution='normal', random_state=0).fit(Xtr)
    return np.clip(q.transform(Xtr), -4, 4), np.clip(q.transform(Xte), -4, 4)


def spline_basis(Xtr, Xte, knots=6):
    """additive cubic spline basis of the rank of every descriptor (uniform scores, so knots sit at quantiles)"""
    from sklearn.preprocessing import QuantileTransformer, SplineTransformer
    q = QuantileTransformer(n_quantiles=min(1000, len(Xtr)), output_distribution='uniform', random_state=0).fit(Xtr)
    s = SplineTransformer(n_knots=knots, degree=3, include_bias=False).fit(q.transform(Xtr))
    return s.transform(q.transform(Xtr)), s.transform(q.transform(Xte))


@reg('ridge')
def ridge(D, tr, te):
    """ridge on rank-normal descriptors, penalty by generalised cross-validation on the training rows"""
    from sklearn.linear_model import RidgeCV
    a, b = rank_normal(D.X[tr], D.X[te])
    return RidgeCV(alphas=np.logspace(-1, 4, 16)).fit(a, D.u[tr], sample_weight=D.w[tr]).predict(b)


@reg('gam')
def gam(D, tr, te):
    """additive model: ridge on a spline basis of every descriptor's rank"""
    from sklearn.linear_model import RidgeCV
    a, b = spline_basis(D.X[tr], D.X[te])
    return RidgeCV(alphas=np.logspace(-1, 4, 16)).fit(a, D.u[tr], sample_weight=D.w[tr]).predict(b)


def soft_rows(X, u, w):
    """rows for a cross-entropy fit with a fractional label u: (x, 1) with weight w u and (x, 0) with weight w (1 - u)"""
    pos, neg = u > 0, u < 1
    return (np.concatenate([X[pos], X[neg]]), np.concatenate([np.ones(pos.sum(), int), np.zeros(neg.sum(), int)]),
            np.concatenate([(w * u)[pos], (w * (1 - u))[neg]]))


@reg('gam_logit')
def gam_logit(D, tr, te):
    """additive logistic model with the fractional label u (spline basis of the ranks)"""
    from sklearn.linear_model import LogisticRegression
    a, b = spline_basis(D.X[tr], D.X[te])
    X2, y2, w2 = soft_rows(a, D.u[tr], D.w[tr])
    return LogisticRegression(C=0.05, max_iter=2000).fit(X2, y2, sample_weight=w2).predict_proba(b)[:, 1]


@reg('hgb_ll')
def hgb_ll(D, tr, te):
    """K1's boosting settings with cross-entropy on the fractional label u instead of squared error"""
    X2, y2, w2 = soft_rows(D.X[tr], D.u[tr], D.w[tr])
    return HistGradientBoostingClassifier(**keep.GBM).fit(X2, y2, sample_weight=w2).predict_proba(D.X[te])[:, 1]


@reg('two_part')
def two_part(D, tr, te):
    """P(a record matches the seed) x E[IoU | matched]; both with K1's boosting settings"""
    X, R, tp, s = D.X[tr], D.w[tr], D.tp[tr].astype(np.float64), D.s_sum[tr]
    pos, neg = tp > 0, tp < R
    Xc = np.concatenate([X[pos], X[neg]])
    yc = np.concatenate([np.ones(pos.sum(), int), np.zeros(neg.sum(), int)])
    wc = np.concatenate([tp[pos], (R - tp)[neg]])
    p = HistGradientBoostingClassifier(**keep.GBM).fit(Xc, yc, sample_weight=wc).predict_proba(D.X[te])[:, 1]
    m = hgb(X[pos], s[pos] / tp[pos], tp[pos], D.X[te])
    return p * m


@reg('mlp')
def mlp(D, tr, te):
    """mean of 5 small networks on rank-normal descriptors"""
    from sklearn.neural_network import MLPRegressor
    a, b = rank_normal(D.X[tr], D.X[te])
    out = 0
    for s in range(5):
        m = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1.0, learning_rate_init=2e-3, max_iter=300, early_stopping=False,
                         random_state=seed_of(tr, 'mlp', s) % 100000)
        out = out + m.fit(a, D.u[tr], sample_weight=D.w[tr]).predict(b)
    return out / 5


# ------------------------------------------------------------------------------------- variance reduction around K1

def image_bootstrap(D, tr, rng):
    """rows of a bootstrap sample of the training IMAGES (an image drawn twice has its seeds twice)"""
    ids = np.unique(D.img[tr])
    count = np.bincount(rng.choice(ids, len(ids), replace=True), minlength=len(D.images))
    return np.repeat(tr, count[D.img[tr]])


def bag(D, tr, te, n_bags, tag, **kw):
    rng = np.random.default_rng(seed_of(tr, tag))
    out = 0
    for _ in range(n_bags):
        b = image_bootstrap(D, tr, rng)
        out = out + hgb(D.X[b], D.u[b], D.w[b], D.X[te], **kw)
    return out / n_bags


@reg('bag10')
def bag10(D, tr, te):
    """K1's model on 10 bootstrap samples of the training images, averaged"""
    return bag(D, tr, te, 10, 'bag10')


@reg('rs5')
def rs5(D, tr, te):
    """5 x K1's model with a random half of the descriptors offered at every split, averaged"""
    out = 0
    for s in range(5):
        out = out + hgb(D.X[tr], D.u[tr], D.w[tr], D.X[te], max_features=0.5, random_state=seed_of(tr, 'rs5', s) % 100000)
    return out / 5


IMAGE_LEVEL = [keep.NAMES.index(n) for n in ('n_seeds', 'img_mean_mean', 'img_seed_area', 'img_p_frac')]


@reg('k1_q8')
def k1_q8(D, tr, te):
    """K1 with the four image-level descriptors cut into 8 quantile bins, so that a tree cannot single out an image"""
    X = D.X.copy()
    for c in IMAGE_LEVEL:
        edges = np.quantile(D.X[tr, c], np.arange(1, 8) / 8)
        X[:, c] = np.searchsorted(edges, D.X[:, c])
    return hgb(X[tr], D.u[tr], D.w[tr], X[te])


MONO = np.zeros(len(keep.NAMES), int)
for _n in ('mean', 'p_in', 'p_p10', 'p_hi5', 'p_hi6', 'p_hi7', 'p_hi8', 'p_hi9', 'mem_a', 'mem_b', 'mem_min', 'contrast', 'contrast2'):
    MONO[keep.NAMES.index(_n)] = 1
MONO[keep.NAMES.index('rank_mean')] = -1


@reg('k1_mono')
def k1_mono(D, tr, te):
    """K1 with monotone constraints: more probability / more contrast never lowers the score"""
    return hgb(D.X[tr], D.u[tr], D.w[tr], D.X[te], monotonic_cst=MONO)


# ------------------------------------------------------------------------------------------------ other targets

TOUCH = 0.1          # a record "drew something on the seed" when its best IoU with the seed exceeds this


@reg('hurdle')
def hurdle(D, tr, te):
    """P(a record drew something on the seed) x P(IoU > 0.5 | drew) x E[IoU | matched], three K1-style models"""
    X, iou = D.X[tr], D.iou[tr]
    R = (iou >= 0).sum(1).astype(np.float64)
    n_touch = (iou > TOUCH).sum(1).astype(np.float64)
    tp = (iou > .5).sum(1).astype(np.float64)
    s = np.where(iou > .5, iou, 0).sum(1)

    def binom(pos_w, neg_w):
        a, b = pos_w > 0, neg_w > 0
        return HistGradientBoostingClassifier(**keep.GBM).fit(
            np.concatenate([X[a], X[b]]), np.concatenate([np.ones(a.sum(), int), np.zeros(b.sum(), int)]),
            sample_weight=np.concatenate([pos_w[a], neg_w[b]])).predict_proba(D.X[te])[:, 1]

    p_touch = binom(n_touch, R - n_touch)
    p_hit = binom(tp, n_touch - tp)
    m = hgb(X[tp > 0], (s / np.maximum(tp, 1))[tp > 0], tp[tp > 0], D.X[te])
    return p_touch * p_hit * m


@reg('raw_iso')
def raw_iso(D, tr, te):
    """Model the mean RAW IoU (not cut at 0.5), then map it to u by an isotonic fit on predictions that are
    cross-fitted by image inside the training rows."""
    from sklearn.isotonic import IsotonicRegression
    iou = D.iou
    raw = np.where(iou >= 0, iou, 0).sum(1) / D.w
    ids = np.unique(D.img[tr])
    rng = np.random.default_rng(seed_of(tr, 'raw_iso'))
    part = dict(zip(ids, rng.permutation(len(ids)) % 5))
    f = np.array([part[i] for i in D.img[tr]])
    inner = np.zeros(len(tr))
    for k in range(5):
        inner[f == k] = hgb(D.X[tr][f != k], raw[tr][f != k], D.w[tr][f != k], D.X[tr][f == k])
    iso = IsotonicRegression(out_of_bounds='clip').fit(inner, D.u[tr], sample_weight=D.w[tr])
    return iso.predict(hgb(D.X[tr], raw[tr], D.w[tr], D.X[te]))


# ----------------------------------------------------------------------------- extra descriptors and refined target

def load_extra(D):
    if 'E' not in D.extra:
        from xk import HERE, target
        with np.load(HERE / 'extra.npz', allow_pickle=False) as z:
            assert np.array_equal(z['fname'].astype(str), D.fname) and np.array_equal(z['label'], D.label)
            D.extra.update(E=z['E'], e_names=[str(x) for x in z['e_names']], iou_ref=z['iou_ref'])
        D.extra['u_ref'], _, D.extra['s_ref'], D.extra['tp_ref'] = target(D.extra['iou_ref'])
        D.extra['XE'] = np.concatenate([D.X, D.extra['E']], 1)
    return D.extra


@reg('k1x')
def k1x(D, tr, te):
    """K1's model on the 50 descriptors plus the extra crop descriptors of build_extra.py"""
    XE = load_extra(D)['XE']
    return hgb(XE[tr], D.u[tr], D.w[tr], XE[te])


@reg('k1_ref')
def k1_ref(D, tr, te):
    """K1's model trained on the u of the FINAL (refined) shape"""
    return hgb(D.X[tr], load_extra(D)['u_ref'][tr], D.w[tr], D.X[te])


def load_r66(D):
    """the 66 descriptors of the independent re-implementation (scratch/ws/k1_reimpl/feats.py: image, the two maps
    and the seeds of the same image only), aligned to the keep tables by seed order within the image"""
    if 'R66' not in D.extra:
        from xk import HERE
        with np.load(HERE / 'reimpl66.npz', allow_pickle=False) as z:
            assert np.array_equal(z['fname'].astype(str), D.fname) and np.array_equal(z['label'], D.label)
            D.extra['R66'] = z['X']
        D.extra['XR'] = np.concatenate([D.X, D.extra['R66']], 1)
    return D.extra


@reg('r66')
def r66(D, tr, te):
    """K1's model on the re-implementation's 66 descriptors instead of the 50"""
    X = load_r66(D)['R66']
    return hgb(X[tr], D.u[tr], D.w[tr], X[te])


@reg('k1r')
def k1r(D, tr, te):
    """K1's model on both descriptor sets together (116)"""
    X = load_r66(D)['XR']
    return hgb(X[tr], D.u[tr], D.w[tr], X[te])


@reg('rf_r66')
def rf_r66(D, tr, te):
    X = load_r66(D)['R66']
    m = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, max_features=1 / 3, n_jobs=JOBS, random_state=seed_of(tr, 'rf_r66'))
    return m.fit(X[tr], D.u[tr], sample_weight=D.w[tr]).predict(X[te])


# --------------------------------------------------------------- several targets at once (the labels say more than u)

def multi_targets(D):
    """per seed: u, the mean raw IoU, the share of records that drew something on it, the share that match it"""
    if 'MT' not in D.extra:
        iou = D.iou
        D.extra['MT'] = np.stack([D.u, np.where(iou >= 0, iou, 0).sum(1) / D.w, (iou > TOUCH).sum(1) / D.w, (iou > .5).sum(1) / D.w], 1)
    return D.extra['MT']


@reg('rf_mt')
def rf_mt(D, tr, te):
    """random forest whose splits serve four targets at once; the first output (u) is the score"""
    Y = multi_targets(D)
    m = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, max_features=1 / 3, n_jobs=JOBS, random_state=seed_of(tr, 'rf_mt'))
    return m.fit(D.X[tr], Y[tr], sample_weight=D.w[tr]).predict(D.X[te])[:, 0]


@reg('et_mt')
def et_mt(D, tr, te):
    Y = multi_targets(D)
    m = ExtraTreesRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.5, n_jobs=JOBS, random_state=seed_of(tr, 'et_mt'))
    return m.fit(D.X[tr], Y[tr], sample_weight=D.w[tr]).predict(D.X[te])[:, 0]


@reg('mlp_mt')
def mlp_mt(D, tr, te):
    """mean of 5 small networks with four outputs (u, raw IoU, drew, matched); the first output is the score"""
    from sklearn.neural_network import MLPRegressor
    a, b = rank_normal(D.X[tr], D.X[te])
    Y = multi_targets(D)
    out = 0
    for s in range(5):
        m = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1.0, learning_rate_init=2e-3, max_iter=300, early_stopping=False,
                         random_state=seed_of(tr, 'mlp_mt', s) % 100000)
        out = out + m.fit(a, Y[tr], sample_weight=D.w[tr]).predict(b)[:, 0]
    return out / 5


# --------------------------------------------------------------------------------------------- more small networks

MLP = dict(hidden_layer_sizes=(64, 32), alpha=1.0, learning_rate_init=2e-3, max_iter=300, early_stopping=False)


def mlp_mean(a, y, w, b, tag, tr, n=5, col=None):
    from sklearn.neural_network import MLPRegressor
    out = 0
    for s in range(n):
        p = MLPRegressor(random_state=seed_of(tr, tag, s) % 100000, **MLP).fit(a, y, sample_weight=w).predict(b)
        out = out + (p if col is None else p[:, col])
    return out / n


@reg('mlp_b')
def mlp_b(D, tr, te):
    """a second, independent set of 5 networks (other random starts): mlp + mlp_b = 10 networks"""
    a, b = rank_normal(D.X[tr], D.X[te])
    return mlp_mean(a, D.u[tr], D.w[tr], b, 'mlp_b', tr)


@reg('mlp_ll')
def mlp_ll(D, tr, te):
    """5 networks with cross-entropy on the fractional label u"""
    from sklearn.neural_network import MLPClassifier
    a, b = rank_normal(D.X[tr], D.X[te])
    X2, y2, w2 = soft_rows(a, D.u[tr], D.w[tr])
    out = 0
    for s in range(5):
        out = out + MLPClassifier(random_state=seed_of(tr, 'mlp_ll', s) % 100000, **MLP).fit(X2, y2, sample_weight=w2).predict_proba(b)[:, 1]
    return out / 5


@reg('mlp_r')
def mlp_r(D, tr, te):
    """5 networks on both descriptor sets (116)"""
    X = load_r66(D)['XR']
    a, b = rank_normal(X[tr], X[te])
    return mlp_mean(a, D.u[tr], D.w[tr], b, 'mlp_r', tr)


@reg('mlp_r66')
def mlp_r66(D, tr, te):
    """5 networks on the re-implementation's 66 descriptors"""
    X = load_r66(D)['R66']
    a, b = rank_normal(X[tr], X[te])
    return mlp_mean(a, D.u[tr], D.w[tr], b, 'mlp_r66', tr)


@reg('mlp_x')
def mlp_x(D, tr, te):
    """5 networks on the 50 descriptors plus the extra crop descriptors"""
    X = load_extra(D)['XE']
    a, b = rank_normal(X[tr], X[te])
    return mlp_mean(a, D.u[tr], D.w[tr], b, 'mlp_x', tr)


@reg('svr')
def svr(D, tr, te):
    """kernel ridge regression (RBF) on rank-normal descriptors, another smooth learner"""
    from sklearn.kernel_ridge import KernelRidge
    a, b = rank_normal(D.X[tr], D.X[te])
    a, b = a / np.sqrt(a.shape[1]), b / np.sqrt(a.shape[1])
    return KernelRidge(kernel='rbf', gamma=0.5, alpha=1.0).fit(a, D.u[tr] - 0.35, sample_weight=D.w[tr]).predict(b) + 0.35


@reg('mlp_r66_b')
def mlp_r66_b(D, tr, te):
    """a second, independent set of 5 networks on the 66 descriptors (other random starts)"""
    X = load_r66(D)['R66']
    a, b = rank_normal(X[tr], X[te])
    return mlp_mean(a, D.u[tr], D.w[tr], b, 'mlp_r66_b', tr)
