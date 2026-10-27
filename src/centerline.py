"""Experiment A: auxiliary centerlines derived only from segmentation polygons."""
import numpy as np


def centerline_target(record_ids, annotations, size, radius=1):
    """Union instance skeletons per annotator, then average annotators at model scale.

    annotations is load_annotations()'s polygon-only mapping. Human spine, bbox,
    area and category fields are neither required nor used. Skeletonization is
    per instance so touching instances do not acquire an artificial shared axis.
    """
    import cv2
    from pycocotools import mask as mu
    from skimage.morphology import skeletonize
    if not record_ids or size < 1 or not isinstance(radius, int) or radius < 0:
        raise ValueError('nonempty records, positive size and nonnegative integer radius required')
    acc = np.zeros((size, size), np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    for record in record_ids:
        union = np.zeros((size, size), bool)
        for polygons in annotations[record]:
            scaled = [[coordinate * size / 2048 for coordinate in polygon] for polygon in polygons]
            mask = mu.decode(mu.merge(mu.frPyObjects(scaled, size, size))).astype(bool)
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                continue
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            crop = mask[y0:y1, x0:x1]
            # Pad to make image-edge objects behave like the same shape in the interior.
            line = skeletonize(np.pad(crop, 1))[1:-1, 1:-1].astype(np.uint8)
            if radius:
                line = cv2.dilate(line, kernel)
            union[y0:y1, x0:x1] |= line.astype(bool) & crop
        acc += union
    return acc / len(record_ids)


def centerline_loss(logits, target):
    """Same BCE + soft Dice form as the mask arm; empty crops remain finite."""
    import torch.nn.functional as F
    probability = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = 1 - (2 * (probability * target).sum() + 1) / (probability.sum() + target.sum() + 1)
    return bce + dice
