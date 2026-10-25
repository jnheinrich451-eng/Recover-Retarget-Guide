"""Measurement and input-contract tests; external model outputs are never mocked."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from scripts import probe_m2d_segformer_makassar as probe


class MakassarProbeTests(unittest.TestCase):
    def test_all_surface_labels_merge_before_boundary(self):
        labels = np.zeros((30, 30), dtype=np.uint8)
        labels[5:25, 5:25] = 1
        labels[10:20, 15] = 2
        labels[15, 10:20] = 3
        mask = probe.surface_mask(labels, [1, 2, 3], 4)
        edge = probe.audit.outer_boundary(mask)
        self.assertEqual(mask.sum(), 400)
        self.assertFalse(edge[6:24, 6:24].any())

    def test_non_square_resize_updates_intrinsics_and_inverse(self):
        source = {"analysis_size_wh": [1024, 1024], "K_analysis": [[500, 0, 512], [0, 600, 512], [0, 0, 1]],
                  "native_to_analysis": np.eye(3).tolist()}
        out = probe.geometry(source, [800, 600])
        self.assertAlmostEqual(out["K_model"][0][2], 400)
        self.assertAlmostEqual(out["K_model"][1][2], 300)
        np.testing.assert_allclose(np.array(out["model_to_analysis"]) @ np.array(out["analysis_to_model"]), np.eye(3))

    def test_author_mapping_read_without_executing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.py"
            path.write_text("raise RuntimeError('must not execute')\nALL_CLASSES = ['background', 'road']\n")
            self.assertEqual(probe.supplied_label_names(path), ["background", "road"])

    def test_empty_detection_has_no_fake_perfect_boundary_score(self):
        lines = [{"name": "known", "evidence": "visible", "points": [[2, 3], [20, 3]]}]
        result, *_ = probe.measure(np.zeros((30, 30), np.uint8), [1, 2, 3], ["bg", "road", "solid", "dashed"], lines)
        self.assertEqual(result["trace_to_boundary"]["status"], "empty_target")
        self.assertIsNone(result["trace_to_boundary"]["median_px"])

    def test_ade_label_ids_cannot_silently_enter_four_class_model(self):
        with self.assertRaises(RuntimeError):
            probe.surface_mask(np.full((4, 4), 6, np.uint8), [1, 2, 3], 4)


if __name__ == "__main__":
    unittest.main()
