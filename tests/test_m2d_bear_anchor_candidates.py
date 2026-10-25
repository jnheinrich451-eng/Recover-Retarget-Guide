"""Test selection, archived provenance and actual control inputs, without model mocks."""
import copy
from pathlib import Path
import tempfile
import unittest

from scripts import probe_m2d_bear_anchor_candidates as probe


class BearAnchorCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = probe.ROOT / "measurements/m2d"
        cls.batch = cls.root / "035_bear_prompt_ab_v1/b_spatial"
        if not (cls.batch / "scoring/score_index.json").exists():
            raise unittest.SkipTest("Returned bear observations unavailable")
        cls.evidence = probe.selection.load_evidence(cls.batch / "candidates", cls.batch / "scoring")
        cls.cfg = probe.load_config(probe.DEFAULT)

    def test_actual_ranking_chooses_6_without_asserting_final_appearance(self):
        before = copy.deepcopy(self.evidence)
        decision = probe.rank_candidates(self.evidence)
        self.assertEqual(decision["ranked_seeds"], [6, 5, 7, 4])
        self.assertEqual((decision["selected_seed"], decision["facing"]), (6, "right"))
        self.assertFalse(decision["production_handoff_authorized"])
        self.assertFalse(decision["human_reviewed"])
        self.assertIsNone(decision["final_anchor_selected"])
        self.assertEqual(before, self.evidence)

    def test_failed_profile_is_not_rescued_by_quality(self):
        evidence = copy.deepcopy(self.evidence)
        row = next(r for r in evidence["rows"] if r["seed"] == 6)
        row["profile"]["observations"]["original"]["body_complete"] = False
        row["metrics"]["quality_subject"] = 9999
        result = probe.rank_candidates(evidence)
        self.assertEqual(result["selected_seed"], 5)
        self.assertNotIn(6, result["ranked_seeds"])

    def test_left_facing_remains_eligible_and_style_does_not_affect_rank(self):
        evidence = copy.deepcopy(self.evidence)
        for row in evidence["rows"]:
            row["metrics"]["style_margin"] = -999
        result = probe.rank_candidates(evidence)
        self.assertEqual(result["ranked_seeds"], [6, 5, 7, 4])
        self.assertTrue(next(r for r in result["candidates"] if r["seed"] == 5)["eligible_for_experiment"])

    def test_invalid_measurement_or_all_refused_has_no_forced_winner(self):
        evidence = copy.deepcopy(self.evidence)
        for row in evidence["rows"]:
            row["metrics"]["quality_subject"] = float("nan")
        self.assertIsNone(probe.rank_candidates(evidence)["selected_seed"])
        for row in evidence["rows"]:
            row["mask_status"] = "refused"
        self.assertEqual(probe.rank_candidates(evidence)["ranked_seeds"], [])

    def test_real_preparation_is_repeatable_and_generator_receives_geometry_only(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "anchors"
            manifest = probe.prepare(self.cfg, self.root, output)
            self.assertEqual(manifest, probe.prepare(self.cfg, self.root, output))
            self.assertEqual(manifest, probe.verified(output, self.cfg))
            self.assertFalse((output / "anchor_candidates.json").exists())
            calls = []
            for seed in (6, 5):
                images, call = probe.expected_call(output, manifest, seed, 0)
                self.assertEqual(len(images), 2)
                self.assertEqual(call["modes"], [2, 0])
                self.assertEqual(call["scales"], [0.6, 0.2])
                self.assertEqual(call["control_guidance_end"], [0.5, 0.5])
                self.assertEqual(call["active_steps"], [12, 12])
                self.assertEqual([im.size for im in images], [(1024, 1024)] * 2)
                self.assertEqual(probe.core.file_hash(output / f"inputs/proxy_{seed}/depth.png"),
                                 manifest["cases"][str(seed)]["controls"]["files"]["bear/depth.png"])
                calls.append(call)
            self.assertEqual(calls[0]["prompt"], calls[1]["prompt"])
            self.assertNotEqual(calls[0]["control_pixels_sha256"], calls[1]["control_pixels_sha256"])
            self.assertEqual({k for k in calls[0] if calls[0][k] != calls[1][k]}, {"proxy_seed", "control_pixels_sha256"})
            # A replaced control must fail before inference, not silently reuse old identity.
            path = output / "inputs/proxy_6/edges.png"
            path.write_bytes((output / "inputs/proxy_6/proxy.png").read_bytes())
            with self.assertRaises(RuntimeError):
                probe.verified(output, self.cfg)

    def test_archive_rejects_wrong_handoff_and_missing_geometry(self):
        alignment = self.root / self.cfg["source_alignment"]
        manifest = probe.core.read_json(alignment / "manifest.json")
        old = probe.core.read_json(alignment / "selection.json")
        row = next(r for r in probe.rank_candidates(self.evidence)["candidates"] if r["seed"] == 6)
        wrong = {**row, "facing": "left"}
        with self.assertRaisesRegex(RuntimeError, "facing"):
            probe.read_case(alignment, "right", wrong, manifest, old)
        with self.assertRaisesRegex(RuntimeError, "Missing saved input"):
            probe.checked_file(alignment, "missing/depth.png", "unused")
        with self.assertRaisesRegex(RuntimeError, "escapes"):
            probe.checked_file(alignment, "../missing.png", "unused")


if __name__ == "__main__":
    unittest.main()
