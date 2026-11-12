"""Known-geometry checks, independent of the phone recordings."""
import unittest
import tempfile
from pathlib import Path

import cv2
import numpy as np

from mypylibs.stereo_depth import calculate_depth, create_depth_video


class StereoDepthTests(unittest.TestCase):
    def setUp(self):
        self.left = np.random.default_rng(42).integers(0, 256, (240, 320), dtype=np.uint8)
        self.K = np.array([[240., 0, 160], [0, 240., 120], [0, 0, 1]])
        self.D = np.zeros(5)

    def shifted_depth(self, axis, sign, baseline=2.0):
        # Camera translation -B yields positive disparity: right(x)=left(x+d).
        t = np.zeros(3)
        t[axis] = -sign
        transform = np.float32([[1, 0, -sign * 8 if axis == 0 else 0],
                                [0, 1, -sign * 8 if axis == 1 else 0]])
        right = cv2.warpAffine(self.left, transform, (320, 240))
        return calculate_depth(self.left, right, self.K, self.D, np.eye(3), t,
                               baseline_cm=baseline)

    def test_known_depth_both_axes_and_signs(self):
        for axis in (0, 1):
            for sign in (-1, 1):
                with self.subTest(axis=axis, sign=sign):
                    _, disparity, depth, _ = self.shifted_depth(axis, sign)
                    self.assertGreater(np.isfinite(depth).sum(), 5000)
                    self.assertAlmostEqual(float(np.nanmedian(disparity)), sign * 8, delta=0.15)
                    # Z = f B / d = 240 * 2 / 8 = 60 cm.
                    self.assertAlmostEqual(float(np.nanmedian(depth)), 60, delta=1)

    def test_baseline_scales_depth(self):
        _, _, a, _ = self.shifted_depth(0, 1, 1)
        _, _, b, _ = self.shifted_depth(0, 1, 3)
        common = np.isfinite(a) & np.isfinite(b)
        np.testing.assert_allclose(b[common], a[common] * 3, rtol=1e-5)

    def test_textureless_frames_do_not_produce_depth(self):
        blank = np.full_like(self.left, 100)
        _, _, depth, _ = calculate_depth(blank, blank, self.K, self.D,
                                         np.eye(3), [-1, 0, 0])
        self.assertFalse(np.isfinite(depth).any())

    def test_invalid_baselines_and_zero_translation(self):
        for baseline in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                calculate_depth(self.left, self.left, self.K, self.D, np.eye(3),
                                [-1, 0, 0], baseline_cm=baseline)
        with self.assertRaises(ValueError):
            calculate_depth(self.left, self.left, self.K, self.D, np.eye(3), [0, 0, 0])

    def test_video_preserves_flagged_pairs_and_scale_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "01_01.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"),
                                     12, (320, 240))
            self.assertTrue(writer.isOpened())
            try:
                for offset in (0, 8, 16):
                    image = cv2.warpAffine(self.left, np.float32([[1, 0, -offset], [0, 1, 0]]),
                                           (320, 240))
                    writer.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
            finally:
                writer.release()
            np.savetxt(root / "frames_001_002_R.csv", np.eye(3), delimiter=",")
            np.savetxt(root / "frames_001_002_t.csv", [-1, 0, 0], delimiter=",")
            rows = [{"pair": "frames_001_002", "status": "ACCEPTED"},
                    {"pair": "frames_002_003", "status": "FLAGGED"}]
            result = create_depth_video(video, self.K, self.D, (320, 240), rows, root, root)
            self.assertTrue(result["scale_assumed"])
            self.assertEqual(result["frames"], 2)
            self.assertEqual(result["frames_with_depth"], 1)
            capture = cv2.VideoCapture(str(root / "01-01.mp4"))
            try:
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
                self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 12, places=1)
                self.assertTrue(capture.read()[0])
                self.assertTrue(capture.read()[0])
                self.assertFalse(capture.read()[0])
            finally:
                capture.release()
            depth = np.load(root / "01_01_depth" / "best_depth.npz")["depth_cm"]
            self.assertAlmostEqual(float(np.nanmedian(depth)), 30, delta=1)


if __name__ == "__main__":
    unittest.main()
