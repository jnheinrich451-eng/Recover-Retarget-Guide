"""Known-answer graph tests using label arrays, not simulated model inference."""
import unittest

import numpy as np

from scripts import probe_m2d_ground_occlusion as probe


class GroundOcclusionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = probe.load_config(probe.DEFAULT_CONFIG)
        self.inherited = probe.base.load_config(probe.base.DEFAULT_CONFIG)
        self.names = {0: "grass", 1: "car", 2: "person", 3: "earth", 4: "road", 5: "path", 6: "dirt track", 7: "wall"}

    def split_scene(self):
        labels = np.zeros((60, 140), np.uint8)
        labels[10:50, 5:75] = 4
        labels[5:55, 30:40] = 1
        labels[5:55, 40:45] = 2  # Person connected to the source car.
        labels[:, 90:138] = 4    # Bigger, unrelated road region must not enter.
        initial = np.zeros_like(labels, bool)
        initial[10:50, 45:75] = True
        return labels, initial, labels == 1

    def apply(self, labels, initial, subject, context="route_labels"):
        return probe.group_road(labels, self.names, initial, subject, context, self.inherited, self.cfg)

    def test_restore_road_across_car_and_person_without_top_k_or_subject_fill(self):
        labels, initial, subject = self.split_scene()
        result, bridge, record = self.apply(labels, initial, subject)
        self.assertTrue(result[20, 10])
        self.assertTrue(result[20, 60])
        self.assertFalse(result[:, 90:].any())
        self.assertFalse((result & np.isin(labels, [1, 2])).any())
        np.testing.assert_array_equal(bridge, np.isin(labels, [1, 2]))
        self.assertEqual(record["added_pixels"], 40 * 25)
        self.assertEqual(len(record["selected_component_ids"]), 2)
        self.assertFalse((initial & ~result).any())

    def test_no_crossing_one_pixel_nonroad_gap(self):
        labels, initial, subject = self.split_scene()
        labels[10:50, 29] = 7  # Wall gap between source car and left road.
        result, _, record = self.apply(labels, initial, subject)
        np.testing.assert_array_equal(result, initial)
        self.assertEqual(record["added_pixels"], 0)

    def test_unrelated_occluder_cannot_extend_group_through_the_whole_scene(self):
        labels, initial, subject = self.split_scene()
        labels[5:55, 60:65] = 2  # Another person, separated from the source group.
        initial[:, 60:] = False
        result, bridge, _ = self.apply(labels, initial, subject)
        self.assertTrue(result[20, 10])
        self.assertTrue(result[20, 50])
        self.assertFalse(result[20, 70])
        self.assertFalse(bridge[:, 60:65].any())

    def test_earth_and_unresolved_contexts_are_preserved(self):
        labels, initial, subject = self.split_scene()
        labels[labels == 4] = 3
        for context, seed in (("earth", initial), ("water", np.zeros_like(initial))):
            result, bridge, record = self.apply(labels, seed, subject, context)
            np.testing.assert_array_equal(result, seed)
            self.assertFalse(bridge.any())
            self.assertFalse(record["applied"])

    def test_different_surface_label_is_not_added_on_opposite_side(self):
        labels, initial, subject = self.split_scene()
        labels[10:50, 5:30] = 3
        result, _, record = self.apply(labels, initial, subject)
        np.testing.assert_array_equal(result, initial)
        self.assertEqual(record["added_pixels"], 0)

    def test_invalid_route_seed_is_refused(self):
        labels, initial, subject = self.split_scene()
        initial[0, 0] = True
        with self.assertRaises(RuntimeError):
            self.apply(labels, initial, subject)

    def test_worst_stroke_exposes_missing_boundary_hidden_by_overall_median(self):
        boundary = np.zeros((100, 100), bool)
        boundary[5:95, 90] = True
        lines = [{"name": "missing_left", "evidence": "visible", "points": [[10, 20], [10, 40]]}]
        for i, start in enumerate((20, 40, 60)):
            lines.append({"name": f"right_{i}", "evidence": "visible", "points": [[90, start], [90, start + 10]]})
        _, result = probe.measures(lines, boundary)
        self.assertEqual(result["trace_to_boundary"]["median_px"], 0)
        self.assertEqual(result["worst_visible_stroke_median_px"], 80)
        self.assertEqual(result["per_visible_stroke"][0]["distance"]["median_px"], 80)


if __name__ == "__main__":
    unittest.main()
