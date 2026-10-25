"""Automatic handoff tests on returned model outputs and algebraic contracts.

No external model is mocked. Artificial policy/JSON inputs below test only
decision rules, never stand in for inference or calibration evidence.
"""
from argparse import Namespace
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import yaml

from scripts import m2d_proxy_facing_selection as select
from scripts import probe_m2d_proxy_alignment as probe
from src.anchor.m2b2 import cache


def observation(facing, **changes):
    return {"facing": facing, "view": "side", "body_complete": True, "head_aligned": True, **changes}


class BidirectionalContractTests(unittest.TestCase):
    def test_both_signs_supported_without_altering_original_record(self):
        for face, opposite in (("left", "right"), ("right", "left")):
            record = {"observations": {"original": observation(face), "mirrored": observation(opposite)}}
            before = copy.deepcopy(record)
            result = select.side_profile(record)
            self.assertEqual(result["status"], "supported")
            self.assertEqual(result["observed_facing"], face)
            self.assertEqual(record, before)

    def test_direction_change_does_not_relax_profile_checks(self):
        for changes in ({"view": "oblique"}, {"body_complete": False}, {"head_aligned": False}):
            with self.subTest(changes=changes):
                record = {"observations": {"original": observation("left", **changes),
                                           "mirrored": observation("right", **changes)}}
                self.assertEqual(select.side_profile(record)["status"], "noncompliant")
        for original, mirrored in ((None, None), (observation("left"), observation("left")),
                (observation("left"), observation("right", view="front")),
                (observation("uncertain"), observation("right")),
                (observation("left", body_complete=None), observation("right"))):
            result = select.side_profile({"observations": {"original": original, "mirrored": mirrored}})
            self.assertEqual(result["status"], "uncertain")

    def test_rotation_transports_both_signs_and_wrong_label_reverses_axis(self):
        stage = {"d_src": [.3, .1, 1.], "u_scene": [0., -1., 0.], "n_plane": [0., -1., 0.]}
        checks = [probe.rotation_check(stage, face) for face in ("left", "right")]
        np.testing.assert_allclose(checks[0]["assigned_forward"], checks[1]["assigned_forward"])
        for check in checks:
            self.assertAlmostEqual(check["forward_dot_target"], 1.)
            self.assertAlmostEqual(check["wrong_label_dot_target"], -1.)
            self.assertAlmostEqual(np.linalg.det(check["R_A"]), 1.)
        with self.assertRaises(ValueError):
            probe.rotation_check(stage, "toward")

    def test_facing_changes_existing_cache_identity(self):
        args = ("bear", "profile prompt", [0, 1], "objects", "1.1", "test-version", 5)
        self.assertNotEqual(cache.cache_key(*args, "left", "same-image"), cache.cache_key(*args, "right", "same-image"))


class ReturnedFacingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = probe.ROOT / "measurements/m2d"
        cls.source = cls.root / "035_bear_prompt_ab_v1/b_spatial"
        if not (cls.source / "scoring/score_index.json").exists():
            raise unittest.SkipTest("Returned Colab measurements are not in this checkout")
        cls.evidence = select.load_evidence(cls.source / "candidates", cls.source / "scoring")

    def test_actual_selection_is_left_5_right_6_but_not_appearance_accepted(self):
        decision = select.decide(self.evidence)
        self.assertEqual(decision["status"], "calibration_required")
        self.assertIsNone(decision["selected_seed"])
        self.assertFalse(decision["automatic_reconstruction_authorized"])
        self.assertEqual({k: (r["seed"], r["facing"]) for k, r in decision["geometry_test_candidates"].items()},
                         {"left": (5, "left"), "right": (6, "right")})
        for case in ("left", "right"):
            handoff = select.request(decision, case, "manifest-fixture")
            self.assertFalse(handoff["human_reviewed"])
            self.assertFalse(handoff["appearance_accepted"])
            self.assertFalse(handoff["production_handoff_authorized"])
            self.assertEqual(handoff["purpose"], "facing_alignment_diagnostic")
        with self.assertRaisesRegex(RuntimeError, "No calibrated"):
            select.request(decision, "selected", "manifest-fixture")

    def test_original_arm_uncertain_view_stays_uncertain(self):
        source = self.source.parent / "a_original"
        evidence = select.load_evidence(source / "candidates", source / "scoring")
        row = next(r for r in select.decide(evidence)["candidates"] if r["seed"] == 7)
        self.assertEqual(row["profile"]["status"], "uncertain")
        self.assertFalse(row["geometry_observation_supported"])

    def fixture_policy(self, evidence):
        # Decision-rule fixture only: these are NOT calibrated deployment limits.
        p = select.policy_template(evidence["config"])
        p.update(status="calibrated", model_lock_sha256=evidence["model_lock_sha256"],
                 calibration_evidence=["unit policy fixture; not for deployment"],
                 profile_validation_evidence=["unit profile fixture; not for deployment"])
        p["minimums"] = {k: -1. for k in select.MINIMUMS}
        p["maximums"] = {k: 1. for k in select.MAXIMUMS}
        return p

    def test_calibrated_rule_exports_observed_left_instead_of_hardcoded_right(self):
        evidence = copy.deepcopy(self.evidence)
        # Restrict an actual measured set to its left case for a decision-rule test.
        evidence["rows"] = [r for r in evidence["rows"] if r["seed"] == 5]
        decision = select.decide(evidence, self.fixture_policy(evidence))
        self.assertEqual((decision["selected_seed"], decision["facing"]), (5, "left"))
        self.assertEqual(select.request(decision, "selected", "manifest-fixture")["facing"], "left")
        self.assertTrue(decision["automatic_reconstruction_authorized"])

    def test_all_style_failures_produce_no_accepted_handoff(self):
        p = self.fixture_policy(self.evidence)
        p["minimums"]["style_margin"] = 0.  # Artificial policy case, not a proposed threshold.
        decision = select.decide(self.evidence, p)
        self.assertEqual(decision["status"], "none_passed")
        self.assertIsNone(decision["selected_seed"])
        self.assertTrue(all("below:style_margin" in r["reasons"] for r in decision["candidates"]))
        with self.assertRaises(RuntimeError):
            select.request(decision, "selected", "manifest-fixture")

    def test_quality_ranks_only_after_appearance_eligibility(self):
        p = self.fixture_policy(self.evidence)
        # A deliberate unit-policy boundary in these measured values. This is
        # not calibration and is never written to an experiment policy file.
        p["minimums"]["style_margin"] = -.075
        decision = select.decide(self.evidence, p)
        self.assertEqual((decision["selected_seed"], decision["facing"]), (5, "left"))
        higher_quality = next(r for r in decision["candidates"] if r["seed"] == 6)
        self.assertFalse(higher_quality["eligible"])
        self.assertIn("below:style_margin", higher_quality["reasons"])

    def test_policy_identity_and_calibration_cannot_be_skipped(self):
        for key, value in (("selection_signature", "different"), ("model_lock_sha256", "different"),
                           ("calibration_evidence", []), ("profile_validation_evidence", []),
                           ("accepted_facings", ["right"]), ("require_flip_consistency", False)):
            p = self.fixture_policy(self.evidence)
            p[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                select.decide(self.evidence, p)

    def arguments(self, tmp):
        return Namespace(config=probe.DEFAULT, probe_root=self.root, drive_root=Path(tmp) / "drive",
                         output_dir=Path(tmp) / "result", policy=None)

    def test_selection_resume_and_tampered_handoff_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.arguments(tmp)
            probe.select(args)
            original = {p: p.read_bytes() for p in args.output_dir.rglob("*") if p.is_file()}
            probe.select(args)
            self.assertEqual(original, {p: p.read_bytes() for p in args.output_dir.rglob("*") if p.is_file()})
            request, _, _ = probe.verified_request(args.output_dir, "left")
            self.assertEqual(request["facing"], "left")
            path = args.output_dir / "cases/left/handoff.json"
            path.write_text(json.dumps({**request, "facing": "right"}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Handoff was modified"):
                probe.verified_request(args.output_dir, "left")

    def test_machine_handoff_checks_original_pixels_and_never_invents_human_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.arguments(tmp)
            probe.select(args)
            folder = args.output_dir / "cases/left"
            work = folder / "work"
            copied = {}
            for stage in ("proxies", "masks"):
                relative = f"{stage}/seed_5"
                source = self.source / "candidates" / relative
                shutil.copytree(source, work / relative)
                for file in (work / relative).iterdir():
                    if file.is_file():
                        copied[file.relative_to(work).as_posix()] = probe.core.file_hash(file)
            probe.core.save_json(work / "imported_candidate.json", {
                "source_batch": str(self.source.resolve()), "files": copied,
                "source_manifest_sha256": probe.core.file_hash(self.source / "candidates/manifest.json")})
            cfg = self.evidence["candidate_manifest"]["config"]
            approval = probe.machine_approval(work, folder / "handoff.json", cfg)
            self.assertEqual(approval["facing"], "left")
            self.assertFalse(approval["human_reviewed"])
            self.assertNotIn("entire_bear_mask_reviewed", approval)
            (work / "masks/seed_5/mask.png").write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "Changed completed artifact"):
                probe.machine_approval(work, folder / "handoff.json", cfg)

    def test_source_pixels_and_model_observations_remain_hash_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            scoring = Path(tmp) / "scoring"
            shutil.copytree(self.source / "scoring", scoring)
            path = scoring / "profile/seed_5/original.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Changed completed artifact"):
                select.load_evidence(self.source / "candidates", scoring)

    def test_actual_035_frame_uses_same_desired_heading_for_both_facings(self):
        stage0 = probe.core.read_json(self.source / "candidates/inputs/source_placement.json")["stage0"]
        checks = [probe.rotation_check(stage0, face) for face in ("left", "right")]
        np.testing.assert_allclose(checks[0]["assigned_forward"], checks[1]["assigned_forward"])
        for check in checks:
            self.assertAlmostEqual(check["forward_dot_target"], 1.)
            self.assertAlmostEqual(check["wrong_label_dot_target"], -1.)


if __name__ == "__main__":
    unittest.main()
