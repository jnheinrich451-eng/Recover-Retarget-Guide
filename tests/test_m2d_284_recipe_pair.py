"""Real archived controls, control traces and scores; no external model mocks."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from scripts import probe_m2d_284_recipe_pair as probe
from scripts import score_m2d_284_recipe_pair as score


class PairedRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = probe.ROOT / "measurements/m2d"
        cls.cfg = probe.configuration()
        cls.source = cls.root / cls.cfg["source_experiment"]
        if not (cls.source / "suite_manifest.json").exists():
            raise unittest.SkipTest("Returned cross-clip inputs unavailable")
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name) / "pair"
        cls.m = probe.prepare(cls.cfg, cls.root, cls.output)
        cls.old = probe.parent.verified(cls.source, probe.parent.configuration())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_prepare_preserves_real_controls_geometry_and_scoring(self):
        self.assertEqual(self.m, probe.prepare(self.cfg, self.root, self.output))
        self.assertEqual(self.m, probe.verified(self.output, self.cfg))
        self.assertEqual(len(self.m["files"]), 6)
        for name in self.m["files"]:
            self.assertEqual((self.output / name).read_bytes(), (self.source / name).read_bytes())
        for key in ("settings", "model_revisions", "measurement_models", "ground", "scoring", "thresholds"):
            self.assertEqual(self.m[key], self.old[key])
        spec, old = self.m["cases"]["284_vehicle"], self.old["cases"]["284_vehicle"]
        self.assertEqual({k for k in spec if spec[k] != old[k]}, {"prompt"})
        self.assertEqual(self.m["user_prompt"], self.m["applied_prompt"])
        self.assertEqual(self.m["prompt_suffix"], "")
        self.assertFalse((self.output / "generation").exists())

    def test_only_recipe_differs_within_pairs(self):
        calls = []
        for seed in self.cfg["seeds"]:
            _, full = probe.expected_call(self.output, self.m, seed, "current_full")
            _, early = probe.expected_call(self.output, self.m, seed, "weaker_early")
            calls.append(full)
            self.assertEqual({k for k in full if full[k] != early[k]},
                             {"arm", "scales", "active_steps", "control_guidance_end"})
            self.assertEqual(full["active_steps"], [24, 24])
            self.assertEqual(early["active_steps"], [12, 12])
            self.assertEqual(early["control_guidance_end"], [.5, .5])
            self.assertEqual(full["control_vae_rng_seed"], seed)
            self.assertEqual(early["control_vae_rng_seed"], seed)
            old = probe.parent.expected_call(self.source, self.old, "284_vehicle", seed)[1]
            self.assertEqual({k for k in old if full[k] != old[k]}, {"prompt"})
        for call in calls[1:]:
            self.assertEqual({k for k in call if call[k] != calls[0][k]}, {"seed", "control_vae_rng_seed"})
        with self.assertRaisesRegex(RuntimeError, "Missing generation"):
            probe.records(self.output, self.m, complete=True)

    def test_saved_real_early_full_pairs_and_damaged_noise_or_schedule(self):
        root = self.root / "road_constraint_v1"
        rows = []
        for arm in ("manual_full", "manual_early"):
            path = root / f"generation/284_vehicle/seed_2/{arm}"
            call = probe.core.read_json(path / "call.json")
            self.assertTrue(probe.timing.completed(path, call))
            rows.append({"call": call, "result": probe.core.read_json(path / "result.json"),
                         "runtime": probe.timing.native.verify_provenance(path, root)[1]})
        self.assertTrue(all(probe.pair_checks(*rows).values()))
        # Regression for the first 284 recipe-pair run: its writer saved the
        # actual noise and encoded controls, but omitted this redundant field.
        # Only metadata is removed here; real model results remain untouched.
        missing = deepcopy(rows)
        for row in missing:
            del row["call"]["control_vae_rng_seed"]
        self.assertTrue(all(probe.pair_checks(*missing).values()))
        self.assertNotIn("control_vae_rng_seed", missing[0]["call"])
        self.assertTrue(all(probe.pair_checks(rows[0], missing[1]).values()))
        misleading = deepcopy(rows)
        for row in misleading:
            row["call"]["control_vae_rng_seed"] += 1
        self.assertFalse(probe.pair_checks(*misleading)["control_vae_rng_matches_generation_seed"])
        damaged_encoding = deepcopy(missing[1])
        damaged_encoding["result"]["actual_union_calls"]["2"]["encoded_control_sha256"] = "0" * 64
        self.assertFalse(probe.pair_checks(missing[0], damaged_encoding)["same_encoded_mode_2"])
        damaged = deepcopy(rows[1])
        damaged["call"]["initial_latents_sha256"] = "0" * 64
        self.assertFalse(probe.pair_checks(rows[0], damaged)["same_initial_latents_sha256"])
        damaged = deepcopy(rows[1])
        damaged["result"]["scheduler_sigmas"][0] += .01
        self.assertFalse(probe.pair_checks(rows[0], damaged)["same_scheduler_sigmas"])
        # Observed 12-step release matches the planned early recipe; a full-step
        # expectation must reject that actual model-call trace.
        early = rows[1]
        changed = deepcopy(early["call"])
        changed["active_steps"] = [24, 24]
        with self.assertRaisesRegex(RuntimeError, "Wrong mode, step or scale"):
            probe.timing.check_trace(changed, early["result"]["trace"])

    def test_unchanged_selection_does_not_claim_absolute_quality(self):
        rows, fixture = score.previous.read_case(self.source, self.old, "284_vehicle")
        old = score.previous.decision(rows, fixture, self.old)
        for arm in self.cfg["arms"]:
            result = score.decision(rows, fixture, self.m, arm)
            self.assertEqual(result["candidates"], old["candidates"])
            self.assertEqual(result["selected_seed"], 6)
            self.assertEqual(result["absolute_quality_acceptance"], "not_measured")
            self.assertFalse(result["production_handoff_authorized"])
        failed_fixture = deepcopy(fixture)
        failed_fixture["selftest"]["road"]["passed"] = False
        self.assertIsNone(score.decision(rows, failed_fixture, self.m, "current_full")["selected_seed"])

    def test_damaged_controls_and_overlapping_source_refused(self):
        path = self.output / "inputs/284_vehicle/edges.png"
        before = path.read_bytes()
        try:
            path.write_bytes(b"damaged")
            with self.assertRaisesRegex(RuntimeError, "Changed completed"):
                probe.verified(self.output, self.cfg)
        finally:
            path.write_bytes(before)
        with self.assertRaisesRegex(RuntimeError, "separate scoring"):
            probe.prepare(self.cfg, self.root, self.source)


if __name__ == "__main__":
    unittest.main()
