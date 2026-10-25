"""Actual archived-input checks and numerical selection invariants, no model mocks."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import probe_m2d_bear_anchor_seed_scores as probe
from scripts import score_m2d_bear_anchor_seeds as score


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = probe.ROOT / "measurements/m2d"
        cls.cfg = probe.configuration(probe.DEFAULT)
        if not (cls.root / cls.cfg["source_pair"] / "pair_result.json").exists():
            raise unittest.SkipTest("Returned fixed-generator pair unavailable")

    def test_four_seeds_change_only_seed_and_preserve_saved_bear_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "probe"
            manifest = probe.prepare(self.cfg, self.root, output)
            self.assertEqual(manifest, probe.prepare(self.cfg, self.root, output))
            self.assertEqual(manifest, probe.verified(output, self.cfg))
            calls = [probe.expected_call(output, manifest, s)[1] for s in self.cfg["seeds"]]
            self.assertEqual(calls[1], {k: manifest["baseline_call"][k] for k in calls[1]})
            for call in calls[1:]:
                self.assertEqual({k for k in call if call[k] != calls[0][k]}, {"seed"})
            self.assertEqual(calls[0]["scales"], [0.6, 0.4])
            self.assertEqual(calls[0]["active_steps"], [24, 24])
            source = self.root / self.cfg["source_pair"]
            self.assertEqual((output / "inputs/bear/depth.png").read_bytes(), (source / "inputs/bear/depth.png").read_bytes())
            self.assertEqual(manifest["geometry"]["rendered_area"], 35000)
            self.assertEqual(manifest["thresholds"], {"m2d_e_loc_rel_max": 0.5, "m2d_e_area_max": 0.693})
            self.assertFalse((output / "selection.json").exists())
            with self.assertRaisesRegex(RuntimeError, "Missing generated"):
                probe.generation_records(output, manifest, require_all=True)
            (output / "inputs/bear/edges.png").write_bytes(b"changed input")
            with self.assertRaisesRegex(RuntimeError, "Changed completed"):
                probe.verified(output, self.cfg)

    def test_wrong_known_image_and_overlapping_output_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = copy.deepcopy(self.cfg)
            cfg["fixture"]["image_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "known-image"):
                probe.prepare(cfg, self.root, Path(temp) / "probe")
        with self.assertRaisesRegex(RuntimeError, "separate scoring"):
            probe.prepare(self.cfg, self.root, self.root / self.cfg["source_pair"])


class DecisionInvariants(unittest.TestCase):
    """Policy-only numerical inputs do not stand in for model predictions."""
    def setUp(self):
        self.cfg = probe.configuration(probe.DEFAULT)
        self.manifest = {"config": self.cfg, "thresholds": {"m2d_e_loc_rel_max": 0.5, "m2d_e_area_max": 0.693}}
        self.fixture = {"selftest": {k: {"passed": True} for k in ("subject", "road", "quality")}}

    def row(self, seed):
        return {"seed": seed, "object": {"object_found": True, "E_loc_rel": 0.1, "E_area": 0.1},
                "metrics": {k: 1.0 for k in self.cfg["selection"]["criteria"]},
                "instrument": {"passed": True}, "association": {"status": "selected_experimental"}}

    def test_scores_cannot_rescue_failed_geometry_or_quality_instrument(self):
        invalid = self.row(0)
        invalid["metrics"] = {k: 999 for k in invalid["metrics"]}
        invalid["object"]["E_area"] = 1.0
        valid = self.row(1)
        result = score.decision([invalid, valid], self.fixture, self.manifest)
        self.assertEqual(result["selected_seed"], 1)
        invalid["object"]["E_area"] = 0.1
        invalid["instrument"]["passed"] = False
        self.assertEqual(score.decision([invalid, valid], self.fixture, self.manifest)["selected_seed"], 1)

    def test_fixture_failure_and_all_missing_masks_abstain_without_forced_winner(self):
        fixture = copy.deepcopy(self.fixture)
        fixture["selftest"]["road"]["passed"] = False
        self.assertIsNone(score.decision([self.row(0)], fixture, self.manifest)["selected_seed"])
        row = self.row(0)
        row["object"]["object_found"] = False
        self.assertIsNone(score.decision([row], self.fixture, self.manifest)["selected_seed"])

    def test_deterministic_ties_and_no_production_or_heading_certification(self):
        rows = [self.row(3), self.row(1)]
        before = copy.deepcopy(rows)
        result = score.decision(rows, self.fixture, self.manifest)
        self.assertEqual(result["selected_seed"], 1)
        self.assertEqual(result, score.decision(rows[::-1], self.fixture, self.manifest))
        self.assertEqual(rows, before)
        self.assertFalse(result["production_accepted"])
        self.assertFalse(result["production_handoff_authorized"])
        self.assertIn("No heading angle", result["candidates"][0]["heading_scope"])

    def test_incomplete_road_readout_is_not_a_zero_error(self):
        boundary = {"per_visible_stroke": [{"fixed_query_pixels": 100, "distance": {"p95_px": 0.0}},
                                            {"fixed_query_pixels": 2, "distance": {"p95_px": None}}]}
        self.assertIsNone(score.road_value(boundary, (100, 100)))
        boundary["per_visible_stroke"][1]["distance"]["p95_px"] = 20.0
        self.assertAlmostEqual(score.road_value(boundary, (100, 100)), -20 / np.hypot(100, 100))


if __name__ == "__main__":
    unittest.main()
