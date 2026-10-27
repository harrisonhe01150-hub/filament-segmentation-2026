"""CPU contract checks for A1; no pretrained downloads or competition test data."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from centerline import centerline_loss, centerline_target
from train_seg import augment, build_model, load_annotations, predict
from predict import load_model

torch.set_num_threads(2)
RECT = [256, 512, 1792, 512, 1792, 768, 256, 768]


class CenterlineTests(unittest.TestCase):
    def test_soft_annotation_consensus_and_empty_record(self):
        one = centerline_target([1], {1: [[RECT]]}, 64)
        half = centerline_target([1, 2], {1: [[RECT]], 2: []}, 64)
        self.assertTrue(np.array_equal(half, one / 2))
        self.assertEqual(cv2.connectedComponents(one.astype(np.uint8))[0] - 1, 1)
        ys, xs = np.nonzero(one)
        self.assertTrue(np.all((ys >= 16) & (ys < 24) & (xs >= 8) & (xs < 56)))
        self.assertLess(one.sum(), 48 * 8 / 2)
        empty = centerline_target([2], {2: []}, 64)
        self.assertTrue(np.isfinite(empty).all() and not empty.any())

    def test_instance_and_expert_order_do_not_change_target(self):
        other = [512, 256, 768, 256, 768, 1792, 512, 1792]
        separate = np.maximum(centerline_target([1], {1: [[RECT]]}, 64),
                              centerline_target([1], {1: [[other]]}, 64))
        together = centerline_target([1], {1: [[other], [RECT]]}, 64)
        self.assertTrue(np.array_equal(separate, together))

    def test_supplied_spine_and_metadata_are_ignored(self):
        root = ROOT / 'scratch' / 'a1_synthetic'
        (root / 'train').mkdir(parents=True, exist_ok=True)
        payload = {'images': [{'id': 1, 'file_name': '20260101000000.jpeg'}],
                   'annotations': [{'image_id': 1, 'segmentation': [RECT],
                                    'spine': {'invalid': 'must never be consumed'},
                                    'bbox': None, 'area': -1, 'category_id': 'unused'}]}
        (root / 'train' / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json').write_text(json.dumps(payload))
        recs, anns = load_annotations(root)
        got = centerline_target(recs['20260101000000.jpeg'], anns, 64)
        self.assertTrue(np.array_equal(got, centerline_target([1], {1: [[RECT]]}, 64)))

    def test_same_geometry_for_both_targets(self):
        image = torch.rand(2, 1, 64, 64)
        target = torch.cat([image, image / 2], 1)
        _, cropped = augment(image, target, 32, torch.Generator().manual_seed(2026))
        self.assertTrue(torch.equal(cropped[:, 1], cropped[:, 0] / 2))

    def test_auxiliary_gradient_and_empty_crop_are_finite(self):
        for empty in [False, True]:
            logits = torch.zeros(2, 1, 16, 16, requires_grad=True)
            target = torch.zeros_like(logits)
            if not empty:
                target[:, :, 8, 2:14] = .5
            loss = centerline_loss(logits, target)
            loss.backward()
            self.assertTrue(torch.isfinite(loss) and torch.isfinite(logits.grad).all())
            self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_prediction_ignores_auxiliary_channel(self):
        class Fixed(torch.nn.Module):
            def __init__(self, auxiliary):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.tensor(.2))
                self.auxiliary = auxiliary
            def forward(self, x):
                mask = x[:, :1] * 0 + self.bias
                return mask if self.auxiliary is None else torch.cat([mask, mask * 0 + self.auxiliary], 1)
        image = np.zeros((2, 32, 32), np.uint8)
        reference = predict(Fixed(None), image, 8, fp16=False)
        for auxiliary in [-100., 100.]:
            self.assertTrue(np.array_equal(reference, predict(Fixed(auxiliary), image, 8, fp16=False)))

    def test_real_encoder_backward_and_checkpoint_channels(self):
        for classes in [1, 2]:
            model = build_model('efficientnet-b4', weights=None, classes=classes)
            model.eval()
            x = torch.rand(2, 3, 64, 64)
            out = model(x)
            self.assertEqual(tuple(out.shape), (2, classes, 64, 64))
            target = torch.zeros(2, 1, 64, 64)
            target[:, :, 20:40, 10:50] = 1
            loss = centerline_loss(out[:, :1], target)
            if classes == 2:
                line = torch.zeros_like(target); line[:, :, 30, 10:50] = 1
                loss = loss + .25 * centerline_loss(out[:, 1:2], line)
            loss.backward()
            self.assertTrue(torch.isfinite(model.segmentation_head[0].weight.grad).all())
            self.assertTrue(all(model.segmentation_head[0].weight.grad[i].abs().sum() > 0 for i in range(classes)))
            with patch('torch.load', return_value=model.state_dict()) as read:
                restored = load_model(Path('no-file-read.pt'), 'efficientnet-b4', torch.device('cpu'))
            self.assertTrue(read.call_args.kwargs['weights_only'])
            with torch.inference_mode():
                self.assertTrue(torch.equal(out.detach(), restored(x)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
