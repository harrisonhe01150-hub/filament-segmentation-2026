"""K1 re-implementation, step 2: my own per-seed descriptors.

Only the image, the two probability maps and the seeds of the same image are used (no annotation, no file name).
Writes scratch/ws/k1_reimpl/feats_f<k>.pkl = dict(names=[...], X=float32 (n_seed, n_feat), fname=[...], sid=[...]).
Seed order = order of seeds_f<k>.pkl (images in probs-file order, seeds by label id).
Usage: python feats.py <fold> [<fold> ...]
"""
import sys, pickle, time
from pathlib import Path
import numpy as np
import cv2
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from seeds import load_fold, avg_map, seeds_of, REPO   # noqa: E402

IMG_DIR = REPO / 'data/MAGFiLO_1.0_Kaggle_2026/train/train_images'
H = W = 2048
CX = CY = 1023.5
RSUN = 900.0          # bright limb ring sits at r ~ 900 px
LIMB = 893.0          # pixels inside this radius count as "on disk"
T_LO = 0.4 * 255

_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)
RR = np.hypot(_xx - CX, _yy - CY)
RB = np.round(RR).astype(np.int32)
NB = int(RB.max()) + 1
_order = np.argsort(RB.ravel(), kind='stable')
_bounds = np.searchsorted(RB.ravel()[_order], np.arange(NB + 1))
DISK9 = RR < 0.9 * RSUN
del _yy, _xx


def ell(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


K1, K3, K5, K10, K25 = np.ones((3, 3), np.uint8), ell(3), ell(5), ell(10), ell(25)

NAMES = [
    # size / shape
    'sh_logarea', 'sh_len', 'sh_wid', 'sh_elong', 'sh_fill', 'sh_solidity', 'sh_npieces', 'sh_bigfrac',
    'sh_thick', 'sh_maxthick', 'sh_compact', 'sh_pca',
    # position on the disk
    'po_x', 'po_y', 'po_r', 'po_rmax', 'po_absy', 'po_ang_h', 'po_ang_rad',
    # stage-1 probability inside / around
    'pr_mean', 'pr_peak', 'pr_p10', 'pr_p50', 'pr_p90', 'pr_std', 'pr_f70', 'pr_f90', 'pr_ring3', 'pr_ring10',
    'pr_halo', 'pr_edge',
    # agreement of the two models
    'ag_diff', 'ag_min', 'ag_mad', 'ag_fk', 'ag_fc', 'ag_liou',
    # flat-fielded image inside / around
    'im_in', 'im_p10', 'im_std', 'im_ring', 'im_ringstd', 'im_contrast', 'im_snr',
    # neighbouring seeds
    'ne_dnear', 'ne_n50', 'ne_n150', 'ne_n300', 'ne_area150', 'ne_near_logarea', 'ne_near_mean', 'ne_debris50',
    # rank within the image / counts per image
    'rk_n', 'rk_nfix', 'rk_logtot', 'rk_rmean', 'rk_rmean_f', 'rk_rarea', 'rk_rarea_f', 'rk_area_rel_max',
    'rk_mean_rel_med', 'rk_imgconf', 'rk_cover01',
    # image-level statistics that do not depend on the seeds
    'gl_med', 'gl_ffstd', 'gl_sharp',
]


def flat_field(img):
    """image / radial median profile (limb darkening removed); profile clamped off-disk."""
    v = img.ravel()[_order]
    prof = np.zeros(NB, np.float32)
    for b in range(NB):
        a0, a1 = _bounds[b], _bounds[b + 1]
        if a1 > a0:
            prof[b] = np.median(v[a0:a1])
    prof = ndi.median_filter(prof, size=9, mode='nearest')
    cen = float(np.median(prof[1:300]))
    ff = img / np.maximum(prof[RB], 0.2 * cen + 1e-3)
    return ff, cen


def box(y0, y1, x0, x1, m):
    return max(0, y0 - m), min(H, y1 + m), max(0, x0 - m), min(W, x1 + m)


def image_feats(img, prob, uk, uc, lab, area, mean, peak):
    """img float32 (0..255); prob float32 average (0..255); uk/uc uint8 upsampled maps; lab seeds 1..n."""
    n = len(area)
    X = np.zeros((n, len(NAMES)), np.float32)
    if n == 0:
        return X
    col = {k: i for i, k in enumerate(NAMES)}
    P = prob * np.float32(1 / 255.0)
    ff, cen = flat_field(img)
    frag = (prob >= T_LO) & (lab == 0)            # stage-1 pieces that are not seeds (< 100 px)

    # ---- image-level (seed independent)
    gl_med = cen / 255.0
    gl_ffstd = float(ff[DISK9].std())
    lap = cv2.Laplacian(ff, cv2.CV_32F, ksize=3)
    gl_sharp = float(np.abs(lap[DISK9]).mean())

    # ---- rank / counts
    r_mean = np.argsort(np.argsort(-mean, kind='stable'), kind='stable').astype(np.float32)
    r_area = np.argsort(np.argsort(-area, kind='stable'), kind='stable').astype(np.float32)
    den = max(n - 1, 1)
    nfix = float(((area >= 300) & (mean >= 0.85)).sum())
    cover01 = float((prob >= 0.1 * 255).sum()) / (np.pi * RSUN ** 2)
    imgconf = float((area * mean).sum() / area.sum())

    objs = ndi.find_objects(lab, max_label=n)
    for j in range(1, n + 1):
        sl = objs[j - 1]
        A = float(area[j - 1])
        f = {}
        # ============ small crop (margin 32) ============
        a0, a1, b0, b1 = box(sl[0].start, sl[0].stop, sl[1].start, sl[1].stop, 32)
        L = lab[a0:a1, b0:b1]
        m = L == j
        m8 = m.astype(np.uint8)
        free = L == 0                                  # not inside any seed
        Pc = P[a0:a1, b0:b1]
        Rc = RR[a0:a1, b0:b1]
        ys, xs = np.nonzero(m)
        assert len(ys) == int(A)

        # ---- shape
        pts = np.stack([xs, ys], 1).astype(np.float32)
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
        ln, wd = max(rw, rh) + 1.0, min(rw, rh) + 1.0
        hull = cv2.convexHull(pts)
        hull_area = cv2.contourArea(hull) + 0.5 * cv2.arcLength(hull, True) + 1.0
        npc, _, st, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
        dt = cv2.distanceTransform(m8, cv2.DIST_L2, 3)
        cont, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perim = sum(cv2.arcLength(c, True) for c in cont) + 4.0
        xm, ym = xs.mean(), ys.mean()
        if len(xs) > 2:
            cov = np.cov(np.stack([xs - xm, ys - ym]).astype(np.float64))
            ev, evec = np.linalg.eigh(cov)
            e1 = evec[:, 1]
            pca = float(np.sqrt((ev[0] + 0.08) / (ev[1] + 0.08)))
        else:
            e1 = np.array([1.0, 0.0]); pca = 1.0
        f['sh_logarea'] = np.log(A)
        f['sh_len'] = ln
        f['sh_wid'] = wd
        f['sh_elong'] = ln / wd
        f['sh_fill'] = A / (ln * wd)
        f['sh_solidity'] = min(1.0, A / hull_area)
        f['sh_npieces'] = npc - 1
        f['sh_bigfrac'] = st[1:, cv2.CC_STAT_AREA].max() / A
        f['sh_thick'] = 4.0 * float(dt[m].mean())
        f['sh_maxthick'] = 2.0 * float(dt.max())
        f['sh_compact'] = min(2.0, 4 * np.pi * A / perim ** 2)
        f['sh_pca'] = pca

        # ---- position
        cx, cy = xm + b0, ym + a0
        px, py = (cx - CX) / RSUN, (cy - CY) / RSUN
        pr = float(np.hypot(px, py))
        f['po_x'] = px
        f['po_y'] = py
        f['po_r'] = pr
        f['po_rmax'] = float(Rc[m].max()) / RSUN
        f['po_absy'] = abs(py)
        f['po_ang_h'] = float(np.degrees(np.arctan2(abs(e1[1]), abs(e1[0]))))
        if pr > 1e-6:
            f['po_ang_rad'] = float(np.degrees(np.arccos(min(1.0, abs(e1[0] * px + e1[1] * py) / pr))))
        else:
            f['po_ang_rad'] = 45.0

        # ---- stage-1 probability
        vals = Pc[m]
        p10, p50, p90 = np.percentile(vals, [10, 50, 90])
        d3 = cv2.dilate(m8, K3).astype(bool)
        d10 = cv2.dilate(m8, K10).astype(bool)
        ring3 = d3 & free
        ring10 = d10 & ~d3 & free
        edge = m & ~cv2.erode(m8, K1).astype(bool)
        f['pr_mean'] = float(mean[j - 1])
        f['pr_peak'] = float(peak[j - 1])
        f['pr_p10'] = p10
        f['pr_p50'] = p50
        f['pr_p90'] = p90
        f['pr_std'] = float(vals.std())
        f['pr_f70'] = float((vals >= 0.7).mean())
        f['pr_f90'] = float((vals >= 0.9).mean())
        f['pr_ring3'] = float(Pc[ring3].mean()) if ring3.any() else 0.4
        f['pr_ring10'] = float(Pc[ring10].mean()) if ring10.any() else 0.0
        f['pr_halo'] = float(((Pc >= 0.1) & d10 & free).sum()) / A
        f['pr_edge'] = float(Pc[edge].mean())

        # ---- agreement of the two models
        Kc = uk[a0:a1, b0:b1]
        Cc = uc[a0:a1, b0:b1]
        kv, cv_ = Kc[m].astype(np.float32), Cc[m].astype(np.float32)
        mk, mc = float(kv.mean()) / 255, float(cv_.mean()) / 255
        reg = d10 & (free | m)
        km, cm = (Kc >= T_LO) & reg, (Cc >= T_LO) & reg
        uni = float((km | cm).sum())
        f['ag_diff'] = mk - mc
        f['ag_min'] = min(mk, mc)
        f['ag_mad'] = float(np.abs(kv - cv_).mean()) / 255
        f['ag_fk'] = float((kv >= T_LO).mean())
        f['ag_fc'] = float((cv_ >= T_LO).mean())
        f['ag_liou'] = float((km & cm).sum()) / uni if uni else 0.0

        # ---- flat-fielded image
        Fc = ff[a0:a1, b0:b1]
        d5 = cv2.dilate(m8, K5).astype(bool)
        d25 = cv2.dilate(m8, K25).astype(bool)
        ring = d25 & ~d5 & free & (Pc < 0.1) & (Rc < LIMB)
        if ring.sum() < 20:
            ring = d25 & ~d5 & free
        if ring.sum() < 5:
            ring = d25 & ~m
        if ring.sum() < 5:
            ring = m
        iv, rv = Fc[m], Fc[ring]
        i_in, r_mu, r_sd = float(iv.mean()), float(rv.mean()), float(rv.std())
        f['im_in'] = i_in
        f['im_p10'] = float(np.percentile(iv, 10))
        f['im_std'] = float(iv.std())
        f['im_ring'] = r_mu
        f['im_ringstd'] = r_sd
        f['im_contrast'] = (r_mu - i_in) / max(r_mu, 1e-3)
        f['im_snr'] = float(np.clip((r_mu - i_in) / max(r_sd, 1e-3), -50, 50))

        # ============ large crop (margin 300): neighbours ============
        A0, A1, B0, B1 = box(sl[0].start, sl[0].stop, sl[1].start, sl[1].stop, 300)
        L2 = lab[A0:A1, B0:B1]
        m2 = L2 == j
        dt2 = cv2.distanceTransform((~m2).astype(np.uint8), cv2.DIST_L2, 5)
        oth = (L2 > 0) & ~m2
        d = np.full(n, np.inf, np.float32)
        if oth.any():
            ids = np.unique(L2[oth])
            d[ids - 1] = ndi.minimum(dt2, L2, ids)
        d[j - 1] = np.inf
        near = int(np.argmin(d))
        has = bool(np.isfinite(d[near]) and d[near] <= 300)
        f['ne_dnear'] = float(min(d[near], 300.0))
        f['ne_n50'] = float((d <= 50).sum())
        f['ne_n150'] = float((d <= 150).sum())
        f['ne_n300'] = float((d <= 300).sum())
        f['ne_area150'] = float(np.log1p(area[d <= 150].sum() / A))
        f['ne_near_logarea'] = float(np.log(area[near])) if has else 0.0
        f['ne_near_mean'] = float(mean[near]) if has else 0.0
        f['ne_debris50'] = float((frag[A0:A1, B0:B1] & (dt2 <= 50)).sum()) / A

        # ---- rank / counts
        f['rk_n'] = n
        f['rk_nfix'] = nfix
        f['rk_logtot'] = float(np.log(area.sum()))
        f['rk_rmean'] = r_mean[j - 1]
        f['rk_rmean_f'] = r_mean[j - 1] / den
        f['rk_rarea'] = r_area[j - 1]
        f['rk_rarea_f'] = r_area[j - 1] / den
        f['rk_area_rel_max'] = A / float(area.max())
        f['rk_mean_rel_med'] = float(mean[j - 1] - np.median(mean))
        f['rk_imgconf'] = imgconf
        f['rk_cover01'] = cover01

        # ---- image-level
        f['gl_med'] = gl_med
        f['gl_ffstd'] = gl_ffstd
        f['gl_sharp'] = gl_sharp

        assert set(f) == set(NAMES), set(NAMES) ^ set(f)
        for k_, v_ in f.items():
            X[j - 1, col[k_]] = v_
    return X


def main(folds, limit=None):
    for k in folds:
        t0 = time.time()
        rows = pickle.load(open(HERE / ('seeds_f%d.pkl' % k), 'rb'))
        files, PK, PC = load_fold(k)
        assert files == [r['fname'] for r in rows]
        Xs, fn, sid = [], [], []
        for i, f in enumerate(files[:limit]):
            img = cv2.imread(str(IMG_DIR / f), cv2.IMREAD_GRAYSCALE)
            assert img is not None and img.shape == (H, W), f
            prob, uk, uc = avg_map(PK[i], PC[i])
            lab, feats, _ = seeds_of(prob)
            r = rows[i]
            assert np.array_equal(feats[:, 0], r['area']) and np.array_equal(feats[:, 1], r['mean'])
            assert np.array_equal(np.flatnonzero(lab.ravel()), np.sort(r['pix']))
            X = image_feats(img.astype(np.float32), prob, uk, uc, lab, feats[:, 0], feats[:, 1], feats[:, 2])
            Xs.append(X); fn += [f] * len(X); sid += list(range(len(X)))
            if i % 20 == 0:
                print('fold', k, i, len(files), f, 'seeds', len(X), round(time.time() - t0), 's', flush=True)
        X = np.concatenate(Xs)
        assert np.isfinite(X).all()
        if limit is None:
            pickle.dump(dict(names=NAMES, X=X, fname=fn, sid=sid), open(HERE / ('feats_f%d.pkl' % k), 'wb'), protocol=4)
        print('FOLD', k, X.shape, round(time.time() - t0), 's', flush=True)
        if limit is not None:
            return X, fn, sid


if __name__ == '__main__':
    main([int(a) for a in sys.argv[1:]])
