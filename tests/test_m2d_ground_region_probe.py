"""Known-answer array tests for postprocessing; no mocked segmentation model."""
import copy
import unittest

import numpy as np

from scripts import probe_m2d_ground_region as probe


class GroundRegionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = probe.load_config(probe.DEFAULT_CONFIG)
        self.names = {0: "grass", 1: "car", 2: "earth", 3: "water", 4: "road", 5: "path", 6: "dirt track"}

    def scene(self, clipped=False):
        labels = np.zeros((60, 80), np.uint8)
        labels[15:, 20:40] = 2
        labels[:, 50:] = 2  # A larger, disconnected earth region must not win.
        labels[20:(60 if clipped else 40), 25:35] = 1
        return labels

    def test_select_contact_component_not_largest_and_join_route_fragment(self):
        labels = self.scene()
        labels[45:48, 22:24] = 6
        labels[0:3, 0:3] = 4
        result, subject, near, candidate = probe.recover(labels, self.names, [30, 30], 40, self.cfg)
        self.assertEqual(result["status"], "ground_candidate_recovered")
        self.assertEqual(result["dominant_context"], "earth")
        self.assertTrue(candidate[46, 23])
        self.assertEqual(result["route_label_pixels_in_candidate"], 6)
        self.assertFalse(candidate[:, 50:].any())
        self.assertFalse(candidate[0:3, 0:3].any())
        self.assertFalse((candidate & subject).any())
        self.assertTrue(near.any())

    def test_clipped_subject_uses_visible_sides_without_inventing_below_frame(self):
        result, subject, near, candidate = probe.recover(self.scene(True), self.names, [30, 30], 60, self.cfg)
        self.assertTrue(result["subject_touches_bottom"])
        self.assertEqual(result["status"], "ground_candidate_recovered")
        self.assertTrue(near[-1].any())
        self.assertEqual(candidate.shape, (60, 80))
        self.assertFalse((near & subject).any())

    def test_dominant_water_cannot_be_filtered_away_to_promote_tiny_earth(self):
        labels = self.scene()
        labels[labels == 2] = 3
        labels[41, 26] = 2
        result, _, _, candidate = probe.recover(labels, self.names, [30, 30], 40, self.cfg)
        self.assertEqual(result["dominant_context"], "water")
        self.assertEqual(result["status"], "unresolved")
        self.assertFalse(candidate.any())
        self.assertFalse(result["automatic_road_guidance"])

    def test_open_grass_is_ground_candidate_without_road_claim_or_border_edges(self):
        labels = np.zeros((60, 80), np.uint8)
        labels[20:40, 25:35] = 1
        result, _, _, candidate = probe.recover(labels, self.names, [30, 30], 40, self.cfg)
        self.assertEqual(result["status"], "ground_candidate_recovered")
        self.assertEqual(result["dominant_context"], "grass")
        self.assertEqual(result["route_label_pixels_in_candidate"], 0)
        self.assertEqual(result["road_status"], "unassessed")
        self.assertFalse(result["automatic_road_guidance"])
        _, edge = probe.boundaries(candidate, labels, self.names, self.cfg)
        self.assertFalse(edge.any())

    def test_unknown_subject_contact_and_tied_context_remain_unresolved(self):
        result, _, _, mask = probe.recover(self.scene(), self.names, [0, 0], 40, self.cfg)
        self.assertEqual(result["reason"], "centroid_not_in_supported_subject_label")
        self.assertFalse(mask.any())
        result, _, _, mask = probe.recover(self.scene(), self.names, [30, 30], None, self.cfg)
        self.assertEqual(result["reason"], "source_contact_unobserved")
        self.assertFalse(mask.any())
        self.assertIsNone(probe.dominant({"earth": 10, "grass": 10}))
        self.assertIsNone(probe.dominant({}))

    def test_occluding_subject_outline_removed_from_diagnostic_edges(self):
        labels = self.scene(True)
        _, _, _, candidate = probe.recover(labels, self.names, [30, 30], 60, self.cfg)
        raw, clean = probe.boundaries(candidate, labels, self.names, self.cfg)
        self.assertGreater(int(raw.sum()), int(clean.sum()))
        self.assertFalse(clean[20:, 24:37].any())
        self.assertTrue(clean[20:59, 20].any())
        self.assertFalse(clean[-1].any())

    def test_coordinate_mapping_records_transform_and_refuses_camera_mismatch(self):
        k = [[500., 0, 512.], [0, 600., 512.], [0, 0, 1.]]
        placement = {"clip": "example", "t_a": 4, "source_stats": {
            "u0": 100.5, "v0": 160.5, "v_contact": 200., "contact_observed": True}}
        metadata = {"clip": "example", "frame": 4, "K_analysis": k,
                    "analysis_size_wh": [1024, 1024], "native_to_analysis": np.eye(3).tolist()}
        qc = {"clip": "example", "K_1024": k}
        point, contact, geometry = probe.source_point(placement, metadata, qc, self.cfg)
        np.testing.assert_array_equal(point, [402, 642])
        self.assertEqual(contact, 800)
        self.assertEqual(geometry["source_stats_to_analysis"], [[4, 0, 0], [0, 4, 0], [0, 0, 1]])
        changed = copy.deepcopy(metadata)
        changed["K_analysis"][0][0] += 1
        with self.assertRaises(RuntimeError):
            probe.source_point(placement, changed, qc, self.cfg)


if __name__ == "__main__":
    unittest.main()
