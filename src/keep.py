"""Learned keep rule: which stage-1 seeds are submitted.

The fixed rule keeps a seed when its area is at least 300 px and its mean probability at least 0.85. That looks at
two numbers of the seed alone. Whether an annotator draws a candidate also depends on things the rule cannot see:
where it sits on the disk, how it compares with the other candidates of the same image (annotators skip the weak
ones when an image is crowded), how dark it is against its surroundings, whether the two models agree on it.

Here every seed of at least 100 px gets about 50 descriptors, and a gradient-boosting regressor predicts
    u = mean over the image's annotator records of (IoU if IoU > 0.5 else 0),
the seed's contribution to the numerator of PQ per record. PQ = sum(matched IoU) / (TP + FP/2 + FN/2) is a ratio,
and keeping a seed adds u to the numerator and 1/2 to the denominator (per record), so keeping pays exactly when
u > PQ / 2. One threshold TAU on the predicted u is therefore the whole decision rule.

Out of fold (model for fold k trained on the other folds, TAU chosen on the other folds) this is +0.008 PQ over
the fixed rule on the same seeds and masks; see EXPERIMENTS.md ("K1").

Nothing here uses annotations except `target`, which is only called on training images to make the labels.
"""
import cv2
import numpy as np

import refine

TAU = 0.25                          # keep a seed when its predicted u is above this (about PQ / 2)
GBM = dict(max_iter=300, learning_rate=0.03, max_depth=4, min_samples_leaf=30, l2_regularization=1.0, random_state=0)
NEIGHBOUR_RADIUS = 150.0            # px between seed centres that counts as "near"
NAMES = ('area', 'log_area', 'mean', 'long', 'short', 'aspect', 'scale', 'fill',
         'rho', 'lat', 'lon', 'abs_lat', 'angle', 'tilt',
         'p_in', 'p_std', 'p_p10', 'p_max', 'p_hi5', 'p_hi6', 'p_hi7', 'p_hi8', 'p_hi9', 'p_ring', 'p_ring2',
         'mem_a', 'mem_b', 'mem_diff', 'mem_min', 'mem_pix_diff',
         'img_in', 'img_std', 'img_p10', 'img_min', 'img_ring', 'contrast', 'img_ring2', 'contrast2', 'edge_grad',
         'others', 'others_near', 'nb_dist', 'nb_count', 'nb_area_ratio',
         'n_seeds', 'img_mean_mean', 'rank_mean', 'rank_area', 'img_seed_area', 'img_p_frac')
_K_NEAR = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
_K_FAR = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
_K_EDGE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def image_descriptors(seeds, members):
    """Descriptors of all seeds of one image: float32 (n_seeds, len(NAMES)).

    seeds: the per-seed arrays of one image as refine.image_bank returns them (or the same rows read back from a
    bank file): area, mean, long, short, scale, M, pix and the uint8 crops image, prob, seed, others.
    members: the probability maps of the two member models at 2048 px (uint8).

    The crops are oriented along the seed and scaled to it, so "inside", "ring" (7 crop px around the seed) and
    "far ring" (7 to 20 crop px) follow the seed's own size.
    """
    n = len(seeds['area'])
    X = np.zeros((n, len(NAMES)), np.float32)
    if n == 0:
        return X
    width = refine.IMAGE_SHAPE[1]
    flat = [m.ravel() for m in members]
    avg = (members[0].astype(np.float32) + members[1].astype(np.float32)) / 2
    img_p_frac = float((avg >= 0.4 * 255).mean())
    means, areas = np.asarray(seeds['mean'], np.float64), np.asarray(seeds['area'], np.float64)
    rank_mean = np.argsort(np.argsort(-means)) / n
    rank_area = np.argsort(np.argsort(-areas)) / n
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
        pix = seeds['pix'][i]
        ma, mb = float(flat[0][pix].mean()) / 255, float(flat[1][pix].mean()) / 255
        pix_diff = float(np.abs(flat[0][pix].astype(np.float32) - flat[1][pix].astype(np.float32)).mean()) / 255
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
                ma, mb, abs(ma - mb), min(ma, mb), pix_diff,
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


def save_table(path, X, u, R):
    """The training table of the final model. The model itself is refitted from it at prediction time (a few
    seconds, deterministic), which avoids shipping a pickle tied to one scikit-learn version."""
    np.savez_compressed(path, X=np.asarray(X, np.float32), u=np.asarray(u, np.float32), R=np.asarray(R, np.int8), names=np.array(NAMES))


def load_model(path):
    with np.load(path, allow_pickle=False) as z:
        if tuple(str(x) for x in z['names']) != NAMES:
            raise ValueError(f'{path} was written for other descriptors')
        return fit(z['X'], z['u'], z['R'])
