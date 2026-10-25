"""Policy/IO tests and real saved masks; no substituted external models."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import probe_m2d_proxy_scores as probe


class ProxyScoresTests(unittest.TestCase):
    def setUp(self):
        self.cfg = probe.load_config()

    def row(self, seed, quality):
        # Artificial policy inputs only: these are not claimed model outputs.
        return {"seed": seed, "geometry": {"mask_nonempty": True, "border_fraction": 0.,
                    "secondary_component_fraction": 0.},
                "metrics": {"subject_similarity": .3, "style_margin": .1, "quality_subject": quality,
                            "profile_margin": .1, "facing_margin": .1, "facing": "right", "flip_consistent": True},
                "instrument": {"original_above_strong_blur": True}}

    def policy(self):
        p = probe.policy_template(self.cfg)
        p.update(status="calibrated", model_lock_sha256="test-model-lock", calibration_evidence=["policy unit fixture"])
        p["minimums"] = {k: 0. for k in probe.MINIMUMS}
        p["minimums"]["quality_subject"] = 50.
        p["maximums"] = {k: 0.1 for k in probe.MAXIMUMS}
        return p

    def decide(self, rows, policy=None):
        return probe.decide(rows, self.cfg, policy or self.policy(), "test-model-lock")

    def test_highest_quality_among_passing_only(self):
        rows = [self.row(0, 95.), self.row(1, 70.), self.row(2, 80.)]
        rows[0]["metrics"]["style_margin"] = -.1
        result = self.decide(rows)
        self.assertEqual(result["selected_seed"], 2)
        self.assertEqual(result["diagnostic_quality_order"], [0, 2, 1])
        self.assertIn("below:style_margin", result["candidates"][0]["reasons"])
        self.assertFalse(result["automatic_reconstruction_authorized"])

    def test_subject_and_seed_are_deterministic_ties(self):
        rows = [self.row(3, 70.), self.row(1, 70.), self.row(2, 70.)]
        rows[0]["metrics"]["subject_similarity"] = .2
        self.assertEqual(self.decide(rows)["selected_seed"], 1)
        self.assertEqual(self.decide(list(reversed(rows)))["selected_seed"], 1)

    def test_uncalibrated_only_ranks_never_passes_or_retries(self):
        result = self.decide([self.row(0, 80.)], probe.policy_template(self.cfg))
        self.assertIsNone(result["selected_seed"])
        self.assertEqual(result["diagnostic_quality_order"], [0])
        self.assertEqual(probe.retry_action(result["status"], 0, 2), "stop")

    def test_all_bad_batch_never_passes_by_relative_rank(self):
        result = self.decide([self.row(0, 20.), self.row(1, 30.)])
        self.assertEqual(result["status"], "none_passed")
        self.assertIsNone(result["selected_seed"])
        self.assertEqual(probe.retry_action(result["status"], 0, 2), "regenerate")
        self.assertEqual(probe.retry_action(result["status"], 2, 2), "exhausted")

    def test_bad_measurements_and_flip_failure_reject(self):
        for kind in ("nan", "missing", "blur", "flip", "empty", "profile"):
            with self.subTest(kind=kind):
                row = self.row(0, 90.)
                if kind == "nan":
                    row["metrics"]["subject_similarity"] = float("nan")
                elif kind == "missing":
                    del row["metrics"]["profile_margin"]
                elif kind == "blur":
                    row["instrument"]["original_above_strong_blur"] = False
                elif kind == "flip":
                    row["metrics"]["flip_consistent"] = False
                elif kind == "empty":
                    row["geometry"]["mask_nonempty"] = False
                else:
                    row["metrics"]["profile_margin"] = -.01
                self.assertIsNone(self.decide([row])["selected_seed"])

    def test_universal_instrument_failure_cannot_trigger_regeneration(self):
        rows = [self.row(0, 80.), self.row(1, 90.)]
        for row in rows:
            row["metrics"]["flip_consistent"] = False
        decision = self.decide(rows)
        self.assertEqual(decision["status"], "instrument_check_required")
        self.assertIn("no_candidate_passed_facing_instrument_check", decision["instrument_issues"])
        self.assertEqual(probe.retry_action(decision["status"], 0, 2), "stop")

    def test_policy_must_match_instrument_and_have_complete_limits(self):
        for kind in ("signature", "lock", "empty_evidence", "none", "missing", "bool"):
            with self.subTest(kind=kind):
                p = self.policy()
                if kind == "signature": p["metric_signature"] = "wrong style"
                elif kind == "lock": p["model_lock_sha256"] = "other model"
                elif kind == "empty_evidence": p["calibration_evidence"] = []
                elif kind == "none": p["minimums"]["quality_subject"] = None
                elif kind == "missing": del p["minimums"]["facing_margin"]
                else: p["minimums"]["quality_subject"] = True
                with self.assertRaises(RuntimeError):
                    self.decide([self.row(0, 80.)], p)

    def test_style_and_crop_are_in_policy_signature(self):
        changed = copy.deepcopy(self.cfg)
        changed["style"] = "illustration"
        self.assertNotEqual(probe.metric_signature(changed), probe.metric_signature(self.cfg))
        changed = copy.deepcopy(self.cfg)
        changed["crop_padding_fraction"] = .2
        self.assertNotEqual(probe.metric_signature(changed), probe.metric_signature(self.cfg))

    def test_crop_and_components_on_known_mask(self):
        mask = np.zeros((100, 100), bool)
        mask[30:70, 20:80] = True
        metrics, box = probe.geometry(mask, .1)
        self.assertEqual(box, [14, 26, 86, 74])
        self.assertEqual(metrics["secondary_component_fraction"], 0)
        mask[0, 0] = True
        metrics, _ = probe.geometry(mask, .1)
        self.assertGreater(metrics["border_fraction"], 0)
        self.assertEqual(metrics["components"], 2)
        self.assertAlmostEqual(metrics["secondary_component_fraction"], 1 / 2401)

    def test_input_and_production_paths_are_protected(self):
        source = Path("/probe/bear")
        for output in (source, source / "scores", source.parent, probe.ROOT / "src/new", Path("/drive/cache/new")):
            with self.assertRaises(RuntimeError):
                probe.guard_paths(source, output)
        probe.guard_paths(source, Path("/probe/scores"))

    def test_actual_returned_images_prepare_and_resume(self):
        source = probe.ROOT / "measurements/m2d/035_bear_v2"
        if not (source / "manifest.json").exists():
            self.skipTest("Returned bear data not in this checkout")
        before = {p: probe.core.file_hash(p) for p in source.rglob("*") if p.is_file()}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            first = probe.prepare(source, out, self.cfg)
            second = probe.prepare(source, out, self.cfg)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 4)
            self.assertTrue(all(r["geometry"]["mask_nonempty"] for r in first))
            for r in first:
                self.assertTrue((out / "inputs" / f"seed_{r['seed']}" / "crop.png").exists())
            self.assertEqual(before, {p: probe.core.file_hash(p) for p in before})

    def test_modified_scores_are_not_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            row = self.row(0, 80.)
            probe.save(out / "scores/seed_0.json", row)
            probe.save(out / "score_index.json", {"scores/seed_0.json": probe.core.file_hash(out / "scores/seed_0.json")})
            row["metrics"]["quality_subject"] = 90.
            (out / "scores/seed_0.json").write_text(json.dumps(row))
            with self.assertRaisesRegex(RuntimeError, "changed scoring input"):
                probe.read_rows(out, [{"seed": 0}])


if __name__ == "__main__":
    unittest.main()
