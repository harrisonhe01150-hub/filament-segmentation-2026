"""Stage 2: a per-filament refiner that looks at every stage-1 instance in its own oriented crop.

Why: the dense U-Net (stage 1) is trained on the mean of the annotators' masks, so its probability mixes two
questions, "would an annotator draw this filament at all" and "where would they put its outline". Here every
stage-1 component (a "seed") gets its own small picture, rotated so that the filament lies horizontally and scaled
so that it fits with room to grow at both ends. A U-Net sees the image, the stage-1 probability, the seed and the
other seeds, and predicts the outline that the annotators who drew *this* filament agreed on.

How its output is used (measured out of fold, see scratch/ws/refine/RESULTS.md): pasting the refined outline as
it is makes the score worse, because for a seed that already matches an annotator the small changes of its
outline are noise. Used with care it helps in two ways. Seeds whose refined masks claim the same pixels are
pieces of one drawn filament and become one instance (`assemble`), and the stage-1 mask is changed only where the
refiner is sure (`refined_piece` with a margin). Which seeds are kept is still decided by the stage-1 rule.

Coordinates: pixel centres sit at integer coordinates (the OpenCV convention). The 2x3 matrix `M` of a seed maps
crop coordinates to image coordinates: [x_image, y_image] = M @ [x_crop, y_crop, 1]. `to_crop` and `to_image` are
the only two functions that resample, so cutting a crop and pasting it back always use the same geometry.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fastsweep import components

IMAGE_SHAPE = (2048, 2048)
CROP_W, CROP_H = 384, 128            # crop size in pixels; the long axis of the seed is horizontal
VIEW_LONG, VIEW_SHORT = 2.0, 2.5     # the crop shows this many seed lengths and seed widths ...
MIN_LONG, MIN_SHORT = 64.0, 16.0     # ... of a seed at least this long and wide (image px) ...
MAX_SCALE = 2.0                      # ... and never magnifies more than this
SEED_T, SEED_CLOSE = 0.4, 8          # stage-1 instances: threshold and grouping radius (predict.POST)
SEED_MIN_AREA = 100                  # smaller components are neither refined nor shown as "other seeds"
KEEP_MIN_AREA, KEEP_MIN_MEAN = 300, 0.85   # the stage-1 keep rule (predict.POST); only kept seeds are submitted
ZONE_T = 0.3                         # lowest seed threshold the training augmentation imitates
MATCH_IOU = 0.15                     # a drawn filament is "the same filament" as a seed above this IoU ...
MATCH_COVER = 0.5                    # ... or when one covers more than this share of the other
PLANES = ('image', 'prob', 'seed', 'others', 'zone')   # uint8 crops stored per seed; the network sees the first four
N_INPUT = 4
ENCODER = 'efficientnet-b4'
ORIGINAL = 2.0                       # priority of a seed pasted unchanged: above any refined probability
MIN_SEED_IOU = 0.2                   # a refined mask that agrees less than this with its seed is not trusted
MAX_SUPERSAMPLE = 8                  # finest sampling grid of a crop that shrinks the image (see to_crop)


# ---------------------------------------------------------------------------------------------------------------
# geometry

def seed_frame(xs, ys):
    """Oriented box of a seed's pixels: centre (x, y), unit vector along the long side, long and short extent."""
    box = cv2.boxPoints(cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32)))
    e1, e2 = box[1] - box[0], box[2] - box[1]
    l1, l2 = float(np.hypot(*e1)), float(np.hypot(*e2))
    long_edge, long, short = (e1, l1, l2) if l1 >= l2 else (e2, l2, l1)
    u = long_edge / long if long > 0 else np.array([1.0, 0.0])
    if u[0] < 0 or (u[0] == 0 and u[1] < 0):      # one of the two equivalent directions, always the same one
        u = -u
    return box.mean(0).astype(np.float64), u.astype(np.float64), long, short


def crop_matrix(centre, u, long, short):
    """Matrix M (crop -> image) and scale s (crop px per image px) of a seed.

    The crop shows VIEW_LONG times the seed's length (the margin at both ends is there because being too short
    is the usual failure) and VIEW_SHORT times its width, never magnified more than MAX_SCALE.
    """
    long, short = max(long, MIN_LONG), max(short, MIN_SHORT)
    s = min(CROP_W / (VIEW_LONG * long), CROP_H / (VIEW_SHORT * short), MAX_SCALE)
    v = np.array([-u[1], u[0]])                   # 90 degrees from u, so the crop is a rotation (no mirror)
    A = np.stack([u, v], 1) / s
    t = centre - A @ np.array([(CROP_W - 1) / 2, (CROP_H - 1) / 2])
    return np.concatenate([A, t[:, None]], 1), s


def footprint(M, shape):
    """Image window (x0, y0, x1, y1), end exclusive, that contains the whole crop."""
    corners = np.array([[-1, -1, 1], [CROP_W, -1, 1], [CROP_W, CROP_H, 1], [-1, CROP_H, 1]], np.float64) @ M.T
    x0, y0 = np.floor(corners.min(0)).astype(int)
    x1, y1 = np.ceil(corners.max(0)).astype(int) + 1
    return max(x0, 0), max(y0, 0), min(x1, shape[1]), min(y1, shape[0])


def to_crop(src, M, origin=(0, 0)):
    """Resample an image, or the window of it whose top-left pixel is `origin`, into the crop (bilinear).

    When the crop shrinks the image, every crop pixel is the mean of the k x k finer samples it covers (k = the
    shrink factor rounded up), so a structure thinner than a crop pixel is dimmed instead of lost between samples.
    """
    m = M.copy()
    m[:, 2] -= origin
    k = min(int(np.ceil(np.hypot(*M[:, 0]) - 1e-9)), MAX_SUPERSAMPLE)
    if k > 1:
        m[:, 2] += m[:, :2] @ np.full(2, 0.5 / k - 0.5)       # centre of the first fine sample
        m[:, :2] /= k
    fine = cv2.warpAffine(src, m, (CROP_W * k, CROP_H * k), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return fine if k == 1 else cv2.resize(fine, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)


def to_image(crop, M, window):
    """Resample a crop into the image window (x0, y0, x1, y1) (bilinear); zero outside the crop."""
    x0, y0, x1, y1 = window
    inv = cv2.invertAffineTransform(M)            # image -> crop
    inv[:, 2] += inv[:, :2] @ np.array([x0, y0], np.float64)
    return cv2.warpAffine(crop, inv, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ---------------------------------------------------------------------------------------------------------------
# seeds, inputs and targets

def flat_field(gray):
    """uint8 image with the large-scale brightness (limb darkening, uneven sky) divided out.

    The ratio to a 40 px Gaussian blur is 1 on quiet sun whatever the distance from disk centre; [0.5, 1.5] is
    stretched to the full byte range, which keeps filaments (ratio about 0.7 to 0.9) well away from both ends.
    """
    g = gray.astype(np.float32)
    ratio = g / np.maximum(cv2.GaussianBlur(g, (0, 0), 40), 1.0)
    return np.round((np.clip(ratio, 0.5, 1.5) - 0.5) * 255).astype(np.uint8)


def pixel_lists(labels, n):
    """Flat pixel indices of labels 1..n of a label map, each sorted ascending."""
    flat = labels.ravel()
    pos = np.flatnonzero(flat)
    order = np.argsort(flat[pos], kind='stable')
    ends = np.cumsum(np.bincount(flat[pos], minlength=n + 1))
    return [pos[order[ends[j - 1]:ends[j]]].astype(np.int32) for j in range(1, n + 1)]


def match_filaments(flat, n, area, ids, recs):
    """Which filament of every annotator record is "the same filament" as each seed.

    flat: label map (flattened) with labels 1..n and pixel counts `area`; ids: labels of the seeds;
    recs: [(record id, [flat pixel indices of each filament])].
    Per record the filament with the largest IoU is the candidate; it counts when the IoU exceeds MATCH_IOU or one
    of the two covers more than MATCH_COVER of the other (a seed that is only a piece of a long drawn filament has
    a small IoU but lies inside it, and that is exactly the case the refiner has to learn).
    Returns (match, iou): (len(ids), len(recs)) filament index or -1, and the candidate's IoU.
    """
    match = np.full((len(ids), len(recs)), -1, np.int32)
    best_iou = np.zeros((len(ids), len(recs)), np.float32)
    a = area[ids].astype(np.float64)
    cols = np.arange(len(ids))
    for r, (_, filaments) in enumerate(recs):
        if not filaments or not len(ids):
            continue
        inter = np.stack([np.bincount(flat[ix], minlength=n + 1)[ids] for ix in filaments]).astype(np.float64)
        size = np.array([len(ix) for ix in filaments], np.float64)[:, None]
        iou = inter / (a[None] + size - inter)
        best = iou.argmax(0)
        b_inter, b_iou, b_size = inter[best, cols], iou[best, cols], size[best, 0]
        same = (b_inter > 0) & ((b_iou > MATCH_IOU) | (b_inter > MATCH_COVER * a) | (b_inter > MATCH_COVER * b_size))
        match[same, r] = best[same]
        best_iou[:, r] = b_iou
    return match, best_iou


def image_bank(prob, gray, recs=None):
    """Every seed of one image with its crops, and with its target when annotations are given.

    prob: stage-1 probability at 2048 on the 0-255 scale (uint8, or float32 for an average of models).
    gray: the uint8 image. recs: [(record id, [flat pixel indices of each filament])] or None (test images).

    Returns a dict of per-seed arrays: label (component number at SEED_T), kept (passes the stage-1 keep rule),
    area, mean (mean probability), M, scale, long, short, pix (list of flat pixel indices), one uint8 crop per
    name in PLANES, and with annotations also target (uint8 crop, 255 = every annotator who drew this filament
    marked the pixel), n_match (how many records drew it; 0 = no target) and best_iou (per record, -1 = no such
    record).
    """
    shape = prob.shape
    labels, feats = components(prob, SEED_T, SEED_CLOSE)
    n = len(feats)
    ids = np.flatnonzero(feats[:, 0] >= SEED_MIN_AREA) + 1
    area = np.concatenate([[0], feats[:, 0]]).astype(np.int64)
    pix = pixel_lists(labels, n)
    low, _ = components(prob, ZONE_T, SEED_CLOSE)       # the groups one threshold step lower contain the seeds
    is_seed = np.zeros(n + 1, bool)
    is_seed[ids] = True
    seed_map = is_seed[labels]
    image = flat_field(gray)
    prob8 = prob if prob.dtype == np.uint8 else np.round(prob).astype(np.uint8)
    out = dict(label=ids.astype(np.int32), area=area[ids].astype(np.int32), mean=feats[ids - 1, 1],
               M=np.zeros((len(ids), 2, 3)), scale=np.zeros(len(ids), np.float32),
               long=np.zeros(len(ids), np.float32), short=np.zeros(len(ids), np.float32),
               pix=[pix[j - 1] for j in ids],
               **{name: np.zeros((len(ids), CROP_H, CROP_W), np.uint8) for name in PLANES})
    out['kept'] = (out['area'] >= KEEP_MIN_AREA) & (out['mean'] >= KEEP_MIN_MEAN)
    if recs is not None:
        match, best_iou = match_filaments(labels.ravel(), n, area, ids, recs)
        out['target'] = np.zeros((len(ids), CROP_H, CROP_W), np.uint8)
        out['n_match'] = (match >= 0).sum(1).astype(np.int8)
        out['best_iou'] = np.full((len(ids), 3), -1, np.float32)
        out['best_iou'][:, :len(recs)] = best_iou
    for k, j in enumerate(ids):
        ys, xs = np.divmod(out['pix'][k], shape[1])
        centre, u, long, short = seed_frame(xs, ys)
        M, s = crop_matrix(centre, u, long, short)
        out['M'][k], out['scale'][k], out['long'][k], out['short'][k] = M, s, long, short
        x0, y0, x1, y1 = footprint(M, shape)
        win = np.s_[y0:y1, x0:x1]
        lab = labels[win]
        own = lab == j

        def cut(mask):
            return to_crop(mask.astype(np.uint8) * 255, M, (x0, y0))

        out['image'][k] = to_crop(image, M)
        out['prob'][k] = to_crop(prob8, M)
        out['seed'][k] = cut(own)
        out['others'][k] = cut(seed_map[win] & ~own)
        # zone: where this seed may appear at a seed threshold between ZONE_T and SEED_T or above, i.e. its group
        # at ZONE_T without the pixels of other components (so the zone at SEED_T is the seed itself)
        group = low.ravel()[out['pix'][k][0]]
        out['zone'][k] = cut((low[win] == group) & ((lab == 0) | own))
        if recs is None or out['n_match'][k] == 0:
            continue
        acc = np.zeros((CROP_H, CROP_W), np.float32)
        for r in np.flatnonzero(match[k] >= 0):
            gy, gx = np.divmod(recs[r][1][match[k, r]], shape[1])
            inside = (gx >= x0) & (gx < x1) & (gy >= y0) & (gy < y1)
            drawn = np.zeros(lab.shape, np.uint8)
            drawn[gy[inside] - y0, gx[inside] - x0] = 255
            acc += to_crop(drawn, M, (x0, y0))
        out['target'][k] = np.round(acc / out['n_match'][k])
    return out


# ---------------------------------------------------------------------------------------------------------------
# bank files (written by analysis/refine_bank.py)

def bank_files(folder, fold=None):
    """Bank files of a folder (of one fold only when `fold` is given), in a fixed order."""
    paths = sorted(Path(folder).glob(f'bank_f{"*" if fold is None else fold}_s*of*.npz'))
    if not paths:
        raise FileNotFoundError(f'no bank files for fold {fold} in {folder}')
    return paths


def load_bank(folder, keys, fold=None):
    """Concatenate the requested per-seed arrays of the bank files in `folder`, ordered by (fname, label).

    'pix' comes back as a list of index arrays. Crops are only read when asked for (they are large).
    """
    parts = {k: [] for k in keys}
    fname, label = [], []
    for path in bank_files(folder, fold):
        with np.load(path, allow_pickle=False) as z:
            fname.append(z['fname'])
            label.append(z['label'])
            for k in keys:
                if k == 'pix':
                    flat, ptr = z['pix'], z['pix_ptr']
                    parts[k] += [flat[ptr[i]:ptr[i + 1]] for i in range(len(ptr) - 1)]
                else:
                    parts[k].append(z[k])
    order = np.lexsort((np.concatenate(label), np.concatenate(fname)))
    return {k: [parts[k][i] for i in order] if k == 'pix' else np.concatenate(parts[k])[order] for k in keys}


def load_images(folder, fold=None):
    """Sorted names of all images the bank was built from, including images without any seed."""
    names = []
    for path in bank_files(folder, fold):
        with np.load(path, allow_pickle=False) as z:
            names += [str(x) for x in z['images']]
    return sorted(names)


# ---------------------------------------------------------------------------------------------------------------
# network

def build_refiner(encoder=ENCODER, weights='imagenet'):
    """U-Net with four input planes (image, probability, seed, other seeds) and one output plane (the outline)."""
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder, encoder_weights=weights, in_channels=N_INPUT, classes=1)


def load_refiner(path, encoder=ENCODER, device='cpu'):
    """Refiner from a state dict saved by analysis/refine_train.py (float16 on disk to keep the file small)."""
    import torch
    model = build_refiner(encoder, weights=None)
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    return model.to(device).eval()


def net_input(planes):
    """Float crops (B, >=4, H, W) in [0, 1] -> network input. The flat-fielded image is centred on quiet sun
    (0.5) and stretched so that a filament is about -1; probability and masks stay in [0, 1]."""
    x = planes[:, :N_INPUT].clone()
    x[:, 0] = (x[:, 0] - 0.5) * 4
    return x


def predict_crops(model, planes, batch=64):
    """Refined probability of every crop, averaged over the four flips. planes: uint8 tensor (N, >=4, H, W) on
    the model's device. Returns a uint8 array (N, H, W)."""
    import torch
    dev = next(model.parameters()).device
    out = np.zeros((len(planes), CROP_H, CROP_W), np.uint8)
    model.eval()
    with torch.inference_mode(), torch.autocast(dev.type, enabled=dev.type == 'cuda'):
        for i in range(0, len(planes), batch):
            x = net_input(planes[i:i + batch].float() / 255)
            acc = 0
            for dims in ((), (-1,), (-2,), (-2, -1)):
                acc = acc + torch.flip(model(torch.flip(x, dims)).float().sigmoid(), dims)
            out[i:i + batch] = (acc[:, 0] / 4 * 255).round().byte().cpu().numpy()
    return out


# ---------------------------------------------------------------------------------------------------------------
# from refined crops back to instances

def refined_piece(refined, seed, M, t_r, shape, fill_holes=False, margin=0.0):
    """The refined mask of one seed in image space, or None when the seed should be pasted unchanged.

    refined, seed: uint8 crops. The crop is thresholded at t_r and only the connected pieces that touch the seed
    are kept (the refiner also lights up neighbours, which have their own crops). The refined *probability* is
    resampled to the image and thresholded there, which gives smoother outlines than resampling a binary mask.

    margin says how far the stage-1 mask is trusted: a pixel of the seed is removed only where the refined
    probability is below t_r - margin, and a new pixel is added only where it is above t_r + margin. With 0 the
    refined mask is taken as it is. Why: three of four kept seeds already match an annotator, and for those the
    small changes the refiner makes to the outline lose more than they win; only its confident changes pay
    (out of fold: removing where it says below about 0.25, adding where it says above about 0.95).

    Returns (idx, p): the flat pixel indices of the mask in the image (ascending) and the probability at each.
    """
    prob = refined.astype(np.float32) / 255
    on = seed >= 128
    if margin:
        prob = np.clip(np.where(on, prob + margin, prob - margin), 0, 1)
    hot = prob >= t_r
    n, parts = cv2.connectedComponents(hot.astype(np.uint8), connectivity=8)
    touched = np.zeros(n, bool)
    touched[parts[on]] = True
    touched[0] = False
    chosen = touched[parts]
    inter = np.count_nonzero(chosen & on)
    if inter == 0 or inter < MIN_SEED_IOU * np.count_nonzero(chosen | on):
        return None                               # nothing usable came back: trust stage 1
    prob[hot & ~chosen] = 0
    x0, y0, x1, y1 = footprint(M, shape)
    p = to_image(prob, M, (x0, y0, x1, y1))
    mask = p >= t_r
    if fill_holes:
        filled = ndi.binary_fill_holes(mask)
        p[filled & ~mask] = t_r                   # enclosed pixels join with the lowest priority
        mask = filled
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return (ys + y0) * shape[1] + xs + x0, p[ys, xs]


def assemble(shape, pix, pieces, cover=None, absorb=None, join=None):
    """Instances of one image from the refined pieces of its seeds: at most one per seed, no shared pixels.

    pix: per seed the flat pixel indices of its stage-1 mask; pieces: per seed the result of refined_piece
    (None = paste the stage-1 mask unchanged).
    Returns (labels, areas, unchanged): the label map (0 = background, 1..K), the pixel count of each of the K
    instances, and how many seeds were pasted unchanged.

    Seeds are handled from the largest to the smallest stage-1 mask. Stage 1 often cuts one drawn filament into
    several seeds; two optional rules put such pieces back together instead of leaving the smaller one behind as
    a false positive (None = rule off):
      cover       the refined mask of a seed covers at least this share of the stage-1 pixels of a smaller seed:
                  the smaller seed stops being an instance and its stage-1 pixels join the larger one.
      absorb      more than this share of the refined mask of a seed was already claimed by larger seeds: the
                  seed joins the instance that claimed most of it.
    join: a second set of pieces that is used only by these two rules (default: `pieces`). It lets the freely
    refined masks decide which seeds belong together while the shapes that are written come from a more cautious
    setting (refined_piece with a margin).
    Pixels that two instances both want go to the higher refined probability (the larger seed on ties); a seed
    pasted unchanged keeps all its pixels. A seed that ends with no pixel is dropped.

    Only the pixels of the masks are ever touched, so the cost does not depend on the image size.
    """
    join = pieces if join is None else join
    labels = np.zeros(shape, np.int32)
    flat = labels.ravel()
    flat_p = np.zeros(flat.shape, np.float32)     # priority with which each pixel is held
    claim = np.zeros(flat.shape, np.int32)        # instance that first claimed each pixel with its join mask
    size = np.array([len(x) for x in pix])
    order = sorted(range(len(pix)), key=lambda i: -size[i])
    seed_of = np.zeros(flat.shape, np.int32)      # stage-1 owner of every pixel (seed index + 1), for `cover`
    if cover is not None:
        for i, x in enumerate(pix):
            seed_of[x] = i + 1
    covered = {}                                  # seed index -> number of the instance that covers it
    count = np.zeros(len(pix) + 1, np.int64)      # pixels per instance number (entry 0 is not used)
    touched = []
    unchanged = 0

    def give(idx, number, p):
        """the pixels `idx` now belong to instance `number` with priority p"""
        count[:] -= np.bincount(flat[idx], minlength=len(count))
        count[number] += len(idx)
        flat[idx] = number
        flat_p[idx] = p
        touched.append(idx)

    for number, i in enumerate(order, 1):
        if i in covered:
            continue
        wanted = pix[i] if join[i] is None else join[i][0]
        owner = claim[wanted]
        if join[i] is not None:
            if absorb is not None and np.count_nonzero(owner) > absorb * len(wanted):
                number = np.bincount(owner[owner > 0]).argmax()
            if cover is not None:
                hit = np.bincount(seed_of[wanted], minlength=len(pix) + 1)[1:]
                for j in np.flatnonzero((hit >= cover * size) & (size < size[i])):
                    covered.setdefault(int(j), number)
        claim[wanted[owner == 0]] = number
        if pieces[i] is None:
            give(pix[i], number, ORIGINAL)
            unchanged += 1
            continue
        idx, p = pieces[i]
        won = p > flat_p[idx]
        give(idx[won], number, p[won])
    for j, number in covered.items():
        give(pix[j], number, ORIGINAL)
    present = count[1:] > 0
    if not present.all():
        renumber = np.zeros(len(count), np.int32)
        renumber[1:][present] = np.arange(1, present.sum() + 1)
        idx = np.concatenate(touched)
        flat[idx] = renumber[flat[idx]]
    return labels, count[1:][present], unchanged


def paste_image(shape, seeds, t_r, fill_holes=False, margin=0.0, t_join=None, cover=None, absorb=None):
    """Refined instances of one image; see `refined_piece` and `assemble` for the result and the settings.

    seeds: dicts with pix (flat pixel indices of the stage-1 mask), M, seed (uint8 crop) and refined (uint8 crop
    of the refiner output, or None to paste the stage-1 mask).
    margin: float('inf') keeps the stage-1 shape of every seed; the refiner then only decides which seeds join.
    t_join: threshold of the plain refined masks (no margin) that decide the joins; None = the pasted masks decide.
    """
    def pieces(t, fill, hold):
        if hold == float('inf'):
            return [None] * len(seeds)
        return [None if sd['refined'] is None else refined_piece(sd['refined'], sd['seed'], sd['M'], t, shape, fill, hold)
                for sd in seeds]

    return assemble(shape, [sd['pix'] for sd in seeds], pieces(t_r, fill_holes, margin), cover, absorb,
                    None if t_join is None else pieces(t_join, False, 0.0))


def submission_rows(bank_dir, pred_dir=None, fold=-1, **paste):
    """Submission rows (filament id, RLE counts) for every image of a bank, as src/make_submission.py writes them.

    pred_dir: folder with pred_f<fold>.npz (analysis/refine_train.py); None pastes the stage-1 seeds, which must
    give the fixed-rule submission mask for mask and so tests this path without a refiner.
    paste: the arguments of paste_image after `seeds` (t_r, margin, t_join, absorb, ...).
    Returns (rows, empty); empty lists the images without any filament.
    """
    from pycocotools import mask as mask_util
    bank = load_bank(bank_dir, ['fname', 'label', 'kept', 'M', 'pix', 'seed'], fold)
    crops = None
    if pred_dir is not None:
        with np.load(Path(pred_dir) / f'pred_f{fold}.npz', allow_pickle=False) as z:
            if not (np.array_equal(z['fname'], bank['fname']) and np.array_equal(z['label'], bank['label'])):
                raise ValueError('predictions and bank list different seeds')
            crops = z['pred']
    rows, empty = [], []
    for f in load_images(bank_dir, fold):
        seeds = [dict(pix=bank['pix'][i], M=bank['M'][i], seed=bank['seed'][i], refined=None if crops is None else crops[i])
                 for i in np.flatnonzero((bank['fname'] == f) & bank['kept'])]
        labels, areas, _ = paste_image(IMAGE_SHAPE, seeds, **paste)
        for n in range(1, len(areas) + 1):
            rle = mask_util.encode(np.asfortranarray((labels == n).astype(np.uint8)))
            rows.append((f'{f.rsplit(".", 1)[0]}_{n}', rle['counts'].decode()))
        if len(areas) == 0:
            empty.append(f)
    return rows, empty
