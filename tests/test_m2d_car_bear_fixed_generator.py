"""Use returned controls and observed generation records; no model substitutes."""
import copy
from pathlib import Path
import tempfile
import unittest

from scripts import probe_m2d_car_bear_fixed_generator as probe


class FixedGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = probe.ROOT / "measurements/m2d"
        cls.cfg = probe.load_config(probe.DEFAULT)
        cls.car = cls.root / cls.cfg["source_car"]
        cls.bear = cls.root / cls.cfg["source_bear"]
        if not (cls.bear / "suite_manifest.json").exists():
            raise unittest.SkipTest("Returned bear observations unavailable")
        cls.car_source = probe.core.read_json(cls.car / "source_manifest.json")
        cls.car_call = probe.core.read_json(cls.car / "runs/depth_car_road/seed_1/call.json")
        cls.bm = probe.core.read_json(cls.bear / "suite_manifest.json")

    def test_recipe_recovers_archived_car_settings_and_only_substitutes_subject(self):
        settings, prompts = probe.recipe(self.car_source, self.car_call, self.bm["prompt"], self.cfg)
        self.assertEqual(settings, probe.RECIPE)
        self.assertEqual(prompts["bear"], prompts["car"].replace("a red car driving", "a brown bear walking on all fours", 1))
        self.assertEqual((settings["depth_scale"], settings["edge_scale"]), (0.6, 0.4))

    def test_changed_scene_prompt_and_weaker_historical_schedule_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Scene prompt"):
            probe.recipe(self.car_source, self.car_call, self.bm["prompt"] + ", snow", self.cfg)
        call = copy.deepcopy(self.car_call)
        call["control_guidance_end"] = [0.5, 0.5]
        with self.assertRaisesRegex(RuntimeError, "car call"):
            probe.recipe(self.car_source, call, self.bm["prompt"], self.cfg)

    def test_preparation_preserves_real_controls_and_is_resumable(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "pair"
            manifest = probe.prepare(self.cfg, self.root, output)
            self.assertEqual(manifest, probe.prepare(self.cfg, self.root, output))
            self.assertEqual(manifest, probe.verified(output, self.cfg))
            self.assertEqual((output / "inputs/car/edges.png").read_bytes(), (self.car / "controls/car_road_edges.png").read_bytes())
            self.assertEqual((output / "inputs/bear/depth.png").read_bytes(), (self.bear / "inputs/proxy_6/depth.png").read_bytes())
            calls = [probe.expected_call(output, manifest, s)[1] for s in probe.SUBJECTS]
            self.assertEqual({k for k in calls[0] if calls[0][k] != calls[1][k]},
                             {"subject", "prompt", "control_pixels_sha256"})
            self.assertEqual(calls[0]["active_steps"], [24, 24])
            self.assertEqual(calls[0]["control_guidance_end"], [1.0, 1.0])
            audit = probe.core.read_json(output / "control_audit.json")
            self.assertEqual(audit["different_edge_pixels"], 0)
            self.assertLessEqual(audit["depth_max_difference_uint8"], 1)
            self.assertFalse(manifest["production_handoff_authorized"])
            self.assertFalse((output / "pair_result.json").exists())
            path = output / "inputs/bear/edges.png"
            path.write_bytes((output / "inputs/bear/proxy.png").read_bytes())
            with self.assertRaisesRegex(RuntimeError, "Changed|changed"):
                probe.verified(output, self.cfg)

    def test_pair_audit_detects_noise_schedule_or_runtime_mismatch_from_real_records(self):
        records = []
        for proxy in (6, 5):
            folder = self.bear / f"generation/seed_1/proxy_{proxy}"
            _, runtime = probe.timing.native.verify_provenance(folder, self.bear)
            records.append((probe.core.read_json(folder / "call.json"),
                            probe.core.read_json(folder / "result.json"), runtime))
        self.assertTrue(all(probe.pair_checks(*records).values()))
        changed = copy.deepcopy(records[1])
        changed[0]["initial_latents_sha256"] = "0" * 64
        changed[0]["active_steps"] = [24, 24]
        changed[2]["packages"]["diffusers"] = "different"
        checks = probe.pair_checks(records[0], changed)
        self.assertFalse(checks["same_initial_latents_sha256"])
        self.assertFalse(checks["same_active_steps"])
        self.assertFalse(checks["same_runtime"])

    def test_output_cannot_overwrite_or_nest_inside_an_input(self):
        for destination in (self.car, self.car / "trial", self.root, probe.ROOT / "src"):
            with self.assertRaisesRegex(RuntimeError, "overlaps"):
                probe.safe_output(self.cfg, self.root, destination)


if __name__ == "__main__":
    unittest.main()
