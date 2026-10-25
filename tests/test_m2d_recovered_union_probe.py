"""Known-answer control and ranking tests. No external-model inference is mocked."""
import copy
import unittest

import numpy as np
from PIL import Image

from scripts import probe_m2d_recovered_union as probe
from scripts import score_m2d_recovered_union as score


class RecoveredUnionTests(unittest.TestCase):
    def setUp(self):
        self.cfg, self.settings = probe.load_config(probe.DEFAULT_CONFIG)
        self.th = {"m2d_e_loc_rel_max": 0.5, "m2d_e_area_max": 0.693}

    def row(self, seed, value=0.5):
        return {"seed": seed, "object": {"object_found": True, "E_loc_rel": 0.1, "E_area": 0.1},
                "metrics": {key: value for key in self.cfg["selection"]["criteria"]}}

    def choose(self, rows, quality=True):
        return score.rank_candidates(rows, self.th, self.cfg["selection"], {"passed": quality})

    def test_identical_depth_first_and_explicit_two_control_schedules(self):
        controls = {n: Image.new("RGB", (8, 8), i) for i, n in enumerate(("depth", "manual", "recovered"))}
        manifest = {"cases": {"404_vehicle": {"prompt": "a red car"}}, "model_revisions": {}}
        calls = [probe.expected_call(self.settings, manifest, controls, "404_vehicle", 0, arm) for arm in self.cfg["arms"]]
        self.assertEqual(len({c["control_pixels_sha256"][0] for c in calls}), 1)
        self.assertEqual(calls[0]["modes"], [2])
        for c in calls[1:]:
            self.assertEqual(c["modes"], [2, 0])
            self.assertEqual(c["scales"], [0.6, 0.4])
            self.assertEqual(c["control_guidance_end"], [1.0, 1.0])

    def test_occluded_edges_removed_without_filling_subject(self):
        edge = np.zeros((20, 20), bool); edge[10, :] = True
        visible = np.zeros_like(edge); visible[8:13, 8:13] = True
        out = probe.compose_recovered(edge, visible)
        self.assertEqual(int(out.sum()), 15)
        self.assertFalse((out & visible).any())

    def test_control_stroke_matches_manual_width_without_crossing_occluder(self):
        edge = np.zeros((20, 20), bool); edge[10, 3:18] = True
        forbidden = np.zeros_like(edge); forbidden[:, 8:12] = True
        control = probe.render_edge_control(edge, forbidden, 2)
        self.assertTrue(control[9, 5] and control[11, 5])
        self.assertFalse((control & forbidden).any())

    def test_area_gate_uses_full_rendered_area_instead_of_visible_fragment(self):
        subject = np.zeros((20, 20), bool); subject[5:15, 5:15] = True
        visible = subject.copy(); visible[:, 10:] = False
        result = score.object_stats(subject, visible, (10, 10), 100)
        self.assertEqual(result["E_area"], 0)
        self.assertEqual(result["A_obj_visible_frac"], 50 / 400)
        self.assertEqual(result["A_obj_rendered_frac"], 100 / 400)

    def test_larger_extra_road_is_penalized(self):
        truth = np.zeros((20, 20), bool); truth[5:15, 5:15] = True
        domain = np.ones_like(truth)
        correct = score.road_stats(truth, truth, domain)
        extra = score.road_stats(domain, truth, domain)
        self.assertEqual(correct["road_iou"], 1.0)
        self.assertEqual(extra["road_iou"], 0.25)
        empty = score.road_stats(np.zeros_like(truth), truth, domain)
        self.assertEqual(empty["road_iou"], 0.0)
        self.assertEqual(empty["road_status"], "no_route_labels_detected")

    def test_fixed_occlusion_domain(self):
        truth = np.ones((20, 20), bool)
        predicted = truth.copy(); predicted[5:15, 5:15] = False
        domain = predicted.copy()
        self.assertEqual(score.road_stats(predicted, truth, domain)["road_iou"], 1.0)
        self.assertLess(score.road_stats(predicted, truth, truth)["road_iou"], 1.0)

    def test_wrong_geometry_cannot_win_with_high_quality(self):
        good, bad = self.row(2, 0.4), self.row(0, 0.9)
        bad["object"]["E_loc_rel"] = 1.0
        result = self.choose([bad, good])
        self.assertEqual(result["selected_seed"], 2)
        self.assertIn("outside_m2d_e_loc_rel_max", result["candidates"][0]["reasons"])

    def test_all_missing_objects_abstain(self):
        row = self.row(0); row["object"] = {"object_found": False}
        result = self.choose([row])
        self.assertEqual(result["status"], "abstained")
        self.assertIsNone(result["selected_seed"])

    def test_missing_nonfinite_scores_do_not_become_passes(self):
        for value in (None, float("nan"), float("inf")):
            row = self.row(0); row["metrics"]["quality_margin"] = value
            self.assertIsNone(self.choose([row])["selected_seed"])

    def test_quality_instrument_failure_abstains(self):
        result = self.choose([self.row(0)], quality=False)
        self.assertIsNone(result["selected_seed"])
        self.assertIn("quality_instrument_failed_blur_check", result["candidates"][0]["reasons"])

    def test_rank_ties_order_independence_and_no_production_acceptance(self):
        rows = [self.row(i) for i in [3, 1, 2, 0]]
        result = self.choose(rows)
        self.assertEqual(result, self.choose(list(reversed(rows))))
        self.assertEqual(result["selected_seed"], 0)
        self.assertFalse(result["production_accepted"])

    def test_better_all_metrics_wins_and_inputs_remain_unchanged(self):
        rows = [self.row(0, 0.4), self.row(3, 0.8)]
        original = copy.deepcopy(rows)
        self.assertEqual(self.choose(rows)["selected_seed"], 3)
        self.assertEqual(rows, original)

    def test_observation_crop_bounds_and_empty_mask(self):
        subject = np.zeros((20, 30), bool); subject[2:12, 0:10] = True
        self.assertEqual(score.crop_box(subject, 0.1), (0, 1, 11, 13))
        self.assertEqual(score.crop_box(np.zeros_like(subject), 0.1), (0, 0, 30, 20))


if __name__ == "__main__":
    unittest.main()
