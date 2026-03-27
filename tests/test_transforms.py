import os
import sys
import types
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)


def _fake_cv2_transform(points, matrix):
    pts = np.asarray(points, dtype=np.float32)
    ones = np.ones((*pts.shape[:-1], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=-1)
    transformed = np.matmul(pts_h, matrix.T)
    return transformed.astype(np.float32)


if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.SimpleNamespace(
        transform=_fake_cv2_transform,
        GaussianBlur=lambda image, kernel, sigmaX=0: image,
        warpAffine=lambda image, matrix, size, flags=0, borderMode=0, borderValue=(0, 0, 0): image,
        fillPoly=lambda *args, **kwargs: None,
        cvtColor=lambda image, code: image,
        COLOR_RGB2HSV=0,
        COLOR_HSV2RGB=1,
        INTER_LINEAR=1,
        BORDER_CONSTANT=0,
    )

from lane_det.datasets.transforms import (
    RandomOcclusion,
    build_translate_shear_matrix,
    warp_lane_rows,
)


class TestTransforms(unittest.TestCase):
    def test_random_occlusion_keeps_annotations(self):
        np.random.seed(0)
        image = np.full((32, 64, 3), 128, dtype=np.uint8)
        lanes = np.array([[10.0, 12.0, 14.0]], dtype=np.float32)
        valid_mask = np.array([[1, 1, 1]], dtype=np.uint8)
        h_samples = np.array([8.0, 16.0, 24.0], dtype=np.float32)

        transform = RandomOcclusion(
            p=1.0,
            num_range=(1, 1),
            width_range=(0.25, 0.25),
            height_range=(0.25, 0.25),
        )
        image_out, lanes_out, mask_out, h_out = transform(image, lanes, valid_mask, h_samples)

        self.assertFalse(np.array_equal(image_out, image))
        np.testing.assert_array_equal(lanes_out, lanes)
        np.testing.assert_array_equal(mask_out, valid_mask)
        np.testing.assert_array_equal(h_out, h_samples)

    def test_warp_lane_rows_applies_translate_and_shear(self):
        lanes = np.array([[50.0, 50.0, 50.0]], dtype=np.float32)
        valid_mask = np.array([[1, 1, 1]], dtype=np.uint8)
        h_samples = np.array([20.0, 40.0, 60.0], dtype=np.float32)
        matrix = build_translate_shear_matrix(img_h=100, tx=5.0, ty=10.0, shear_x=0.1)

        warped_lanes, warped_mask = warp_lane_rows(
            lanes=lanes,
            valid_mask=valid_mask,
            h_samples=h_samples,
            matrix=matrix,
            img_w=200,
            img_h=100,
        )

        expected_lanes = np.array([[0.0, 53.05, 55.05]], dtype=np.float32)
        expected_mask = np.array([[0, 1, 1]], dtype=np.uint8)

        np.testing.assert_allclose(warped_lanes, expected_lanes, atol=1e-4)
        np.testing.assert_array_equal(warped_mask, expected_mask)


if __name__ == "__main__":
    unittest.main()
