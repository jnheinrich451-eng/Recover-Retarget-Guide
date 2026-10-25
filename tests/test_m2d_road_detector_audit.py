"""Known-answer tests for boundary measurements; no model simulations."""
import unittest

import numpy as np

from scripts import probe_m2d_road_detector_audit as audit


class RoadDetectorAuditTests(unittest.TestCase):
    def test_empty_mask_has_no_edges_and_is_not_perfect_match(self):
        empty = np.zeros((40, 40), bool)
        query = empty.copy()
        query[20, 20] = True
        self.assertFalse(audit.outer_boundary(empty).any())
        result = audit.distances(query, empty)
        self.assertEqual(result["status"], "empty_target")
        self.assertIsNone(result["median_px"])

    def test_image_border_and_enclosed_hole_are_not_road_edges(self):
        mask = np.ones((40, 40), bool)
        mask[15:25, 15:25] = False
        self.assertFalse(audit.outer_boundary(mask).any())

    def test_known_rectangle_has_only_outer_boundary(self):
        mask = np.zeros((40, 40), bool)
        mask[5:35, 8:32] = True
        mask[15:25, 15:25] = False
        edge = audit.outer_boundary(mask)
        self.assertTrue(edge[5, 8:32].all())
        self.assertTrue(edge[34, 8:32].all())
        self.assertTrue(edge[5:35, 8].all())
        self.assertFalse(edge[14:26, 14:26].any())

    def test_exact_and_shifted_lines_have_known_distance(self):
        a = np.zeros((40, 40), bool)
        b = a.copy()
        a[5:35, 12] = True
        b[5:35, 17] = True
        self.assertEqual(audit.distances(a, a)["mean_px"], 0.0)
        result = audit.distances(a, b)
        self.assertAlmostEqual(result["mean_px"], 5.0)
        self.assertAlmostEqual(result["p95_px"], 5.0)

    def test_inferred_segments_are_excluded(self):
        lines = [{"evidence": "visible", "points": [[5, 10], [15, 10]]},
                 {"evidence": "inferred", "points": [[15, 10], [25, 10]]}]
        raster = audit.visible_traces(lines, (40, 40))
        self.assertEqual(int(raster.sum()), 11)
        self.assertFalse(raster[:, 16:].any())

    def test_trace_outside_image_refused(self):
        with self.assertRaises(RuntimeError):
            audit.visible_traces([{"evidence": "visible", "points": [[0, 0], [40, 10]]}], (40, 40))


if __name__ == "__main__":
    unittest.main()
