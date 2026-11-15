"""Learned keep rule: which stage-1 seeds are submitted.

The fixed rule keeps a seed when its area is at least 300 px and its mean probability at least 0.85. That looks at
two numbers of the seed alone. Whether an annotator draws a candidate also depends on things the rule cannot see:
where it sits on the disk, how it compares with the other candidates of the same image (annotators skip the weak
ones when an image is crowded), how dark it is against its surroundings, whether the member models agree on it.

Here every seed of at least 100 px gets about 50 descriptors, and a gradient-boosting regressor predicts
    u = mean over the image's annotator records of (IoU if IoU > 0.5 else 0),
the seed's contribution to the numerator of PQ per record. PQ = sum(matched IoU) / (TP + FP/2 + FN/2) is a ratio,
and keeping a seed adds u to the numerator and 1/2 to the denominator (per record), so keeping pays exactly when
u > PQ / 2. One threshold TAU on the predicted u is therefore the whole decision rule.

Out of fold (model for fold k trained on the other folds, threshold chosen without fold k) this is about +0.007 PQ
over the fixed rule on the same seeds and masks; see EXPERIMENTS.md ("K1") and analysis/keep_fit.py.

Nothing here uses annotations except `target`, which is only called on training images to make the labels.
"""
import cv2
import numpy as np

import refine

# Keep a seed when its predicted u is above TAU. PQ / 2 of the training folds is about 0.23, and out of fold every
# value from 0.22 to 0.25 scores the same within noise (analysis/keep_fit.py prints the curve). 0.25 is the upper
# end of that range: on the public leaderboard PQ is about 0.88 of the out-of-fold value because fewer predictions
# find a partner in the test annotations, which makes a false positive relatively more costly there.
TAU = 0.25
# early_stopping is switched off explicitly: scikit-learn's default ('auto') would hold out 10 % of the rows as
# soon as a table has more than 10,000 of them, and the fit would silently change.
GBM = dict(max_iter=300, learning_rate=0.03, max_depth=4, min_samples_leaf=30, l2_regularization=1.0,
           early_stopping=False, random_state=0)
NEIGHBOUR_RADIUS = 150.0            # px between seed centres that counts as "near"
# p_crop_max is the maximum over the whole crop (it can be a neighbour's peak); the other p_* are inside the seed
# or in the rings around it. The mem_* descriptors compare the member models and are symmetric in them: the
# result does not depend on the order in which the models are given.
NAMES = ('area', 'log_area', 'mean', 'long', 'short', 'aspect', 'scale', 'fill',
         'rho', 'lat', 'lon', 'abs_lat', 'angle', 'tilt',
         'p_in', 'p_std', 'p_p10', 'p_crop_max', 'p_hi5', 'p_hi6', 'p_hi7', 'p_hi8', 'p_hi9', 'p_ring', 'p_ring2',
         'mem_max', 'mem_min', 'mem_spread', 'mem_mean', 'mem_pix_spread',
         'img_in', 'img_std', 'img_p10', 'img_min', 'img_ring', 'contrast', 'img_ring2', 'contrast2', 'edge_grad',
         'others', 'others_near', 'nb_dist', 'nb_count', 'nb_area_ratio',
         'n_seeds', 'img_mean_mean', 'rank_mean', 'rank_area', 'img_seed_area', 'img_p_frac')
_K_NEAR = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
_K_FAR = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
_K_EDGE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def _rank(values):
    """Position of each value when sorted from largest to smallest; ties keep their order (stable sort), so every
    machine gets the same ranks."""
    return np.argsort(np.argsort(-values, kind='stable'), kind='stable')


def image_descriptors(seeds, members):
    """Descriptors of all seeds of one image: float32 (n_seeds, len(NAMES)).

    seeds: the per-seed arrays of one image as refine.image_bank returns them (or the same rows read back from a
    bank file): area, mean, long, short, scale, M, pix and the uint8 crops image, prob, seed, others.
    members: the probability maps of the member models at 2048 px (uint8), two or more, in any order. Use as many
    members as the training table was built with (keep.load_model checks it).

    The crops are oriented along the seed and scaled to it, so "inside", "ring" (7 crop px around the seed) and
    "far ring" (7 to 20 crop px) follow the seed's own size.
    """
    if len(members) < 2:
        raise ValueError('the descriptors compare at least two member models')
    n = len(seeds['area'])
    X = np.zeros((n, len(NAMES)), np.float32)
    if n == 0:
        return X
    width = refine.IMAGE_SHAPE[1]
    flat = [m.ravel() for m in members]
    avg = np.mean([m.astype(np.float32) for m in members], 0)
    img_p_frac = float((avg >= 0.4 * 255).mean())
    means, areas = np.asarray(seeds['mean'], np.float64), np.asarray(seeds['area'], np.float64)
    rank_mean, rank_area = _rank(means) / n, _rank(areas) / n
    centres = np.array([[(p % width).mean(), (p // width).mean()] for p in seeds['pix']])
    for i in range(n):
        seed = seeds['seed'][i] >= 128
        if not seed.any():                                   # a seed thinner than a crop pixel
            seed = seeds['prob'][i] >= 0.4 * 255
            if not seed.any():
                seed[refine.CROP_H // 2, refine.CROP_W // 2] = True
        s8 = seed.astype(np.uint8)
        near = cv2.dilate(s8, _K_NEAR) > 0
        ring = near & ~seed
        ring2 = (cv2.dilate(s8, _K_FAR) > 0) & ~near
        edge = seed & ~(cv2.erode(s8, _K_EDGE) > 0)
        img, p = seeds['image'][i].astype(np.float32), seeds['prob'][i].astype(np.float32) / 255
        other = seeds['others'][i] >= 128
        M = seeds['M'][i]
        c = M @ np.array([(refine.CROP_W - 1) / 2, (refine.CROP_H - 1) / 2, 1.0])
        lat, lon = (c[1] - 1023.5) / 900, (c[0] - 1023.5) / 900
        angle = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])) % 180)
        tilt = (angle if angle <= 90 else angle - 180) * (1 if lat < 0 else -1)
        v = np.stack([f[seeds['pix'][i]] for f in flat]).astype(np.float32)      # (members, pixels of the seed)
        member_means = v.mean(1) / 255
        mem_max, mem_min = float(member_means.max()), float(member_means.min())
        pix_spread = float((v.max(0) - v.min(0)).mean()) / 255
        grad = np.hypot(cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3))
        d = np.hypot(centres[:, 0] - centres[i, 0], centres[:, 1] - centres[i, 1])
        d[i] = 1e9
        nb = d < NEIGHBOUR_RADIUS
        inside = img[seed].mean()
        ring_m = img[ring].mean() if ring.any() else 127.0
        ring2_m = img[ring2].mean() if ring2.any() else 127.0
        area, long, short = float(seeds['area'][i]), float(seeds['long'][i]), float(seeds['short'][i])
        X[i] = [area, np.log(area), seeds['mean'][i], long, short, long / max(short, 1), seeds['scale'][i], area / max(long * short, 1),
                float(np.hypot(lat, lon)), lat, lon, abs(lat), angle, tilt,
                p[seed].mean(), p[seed].std(), np.percentile(p[seed], 10), p.max(),
                (p[seed] >= .5).mean(), (p[seed] >= .6).mean(), (p[seed] >= .7).mean(), (p[seed] >= .8).mean(), (p[seed] >= .9).mean(),
                p[ring].mean() if ring.any() else 0, p[ring2].mean() if ring2.any() else 0,
                mem_max, mem_min, mem_max - mem_min, float(member_means.mean()), pix_spread,
                inside, img[seed].std(), np.percentile(img[seed], 10), img[seed].min(), ring_m, ring_m - inside,
                ring2_m, ring2_m - inside, grad[edge].mean() if edge.any() else 0,
                other.mean(), (other & ring2).sum() / max(ring2.sum(), 1), min(float(d.min()), 2000.0), int(nb.sum()),
                area / max(areas[nb].max(), 1) if nb.any() else 10.0,
                n, float(means.mean()), rank_mean[i], rank_area[i], float(areas.sum()), img_p_frac]
    return X


def target(best_iou):
    """Training labels from a bank's best_iou (n, 3; -1 where the image has no such record):
    u (mean matched IoU per record), R (number of records), s (sum of matched IoU) and tp (records matched)."""
    iou = np.asarray(best_iou, np.float64)
    R = (iou >= 0).sum(1)
    hit = iou > .5
    s = (iou * hit).sum(1)
    return s / np.maximum(R, 1), R, s, hit.sum(1)


def fit(X, u, R):
    """The keep model: predicts u from the descriptors; a record-weighted regression, because every annotator
    record of an image counts once in the metric."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(**GBM).fit(X, u, sample_weight=np.asarray(R, np.float64))


def save_table(path, X, u, R, members=2):
    """The training table of the final model. The model itself is refitted from it at prediction time (a few
    seconds, deterministic), which avoids shipping a pickle tied to one scikit-learn version.
    members: how many member models the descriptors were computed from."""
    np.savez_compressed(path, X=np.asarray(X, np.float32), u=np.asarray(u, np.float32), R=np.asarray(R, np.int8),
                        names=np.array(NAMES), members=np.int64(members))


def load_model(path, members=None):
    """The keep model refitted from its table. members: the number of member models that will be passed to
    image_descriptors; a table built from another number is refused."""
    with np.load(path, allow_pickle=False) as z:
        if tuple(str(x) for x in z['names']) != NAMES:
            raise ValueError(f'{path} was written for other descriptors')
        if members is not None and int(z['members']) != members:
            raise ValueError(f'{path} was built from {int(z["members"])} member models, not {members}')
        return fit(z['X'], z['u'], z['R'])
