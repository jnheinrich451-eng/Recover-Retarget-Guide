"""CPU instrument/contract tests. No model weights or simulated inference."""
import copy
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from scripts import probe_m2d_284_union_pro as probe


class UnionProProbeTests(unittest.TestCase):
    def test_depth_first_and_both_schedules_are_explicit(self):
        cfg = probe.load_config(probe.DEFAULT_CONFIG)
        depth, road = Image.new("RGB", (1024, 1024), "gray"), Image.new("RGB", (1024, 1024), "black")
        controls = {"depth.png": depth, "road_edges.png": road}
        manifest = {"prompt": "same city prompt", "model_revisions": {"model": "pinned"}}
        left = probe.expected_call(cfg, manifest, controls, 3, "depth")
        right = probe.expected_call(cfg, manifest, controls, 3, "depth_road")
        self.assertEqual(left["modes"], [2])
        self.assertEqual(right["modes"], [2, 0])
        self.assertEqual(left["scales"], [0.6])
        self.assertEqual(right["scales"], [0.6, 0.4])
        self.assertEqual(left["control_guidance_end"], [1.0])
        self.assertEqual(right["control_guidance_end"], [1.0, 1.0])
        self.assertEqual(left["control_pixels_sha256"][0], right["control_pixels_sha256"][0])
        for key in ("prompt", "control_vae_rng_seed", "steps", "guidance", "model_revisions"):
            self.assertEqual(left[key], right[key])
        with self.assertRaises(RuntimeError):
            probe.arm_inputs(cfg, controls, "typo")

    def test_known_recorded_noise_and_depth_pair_then_corrupt_each_field(self):
        # Observed hashes from road_generalization_v1 / 284_vehicle / seed_0,
        # subject vs subject_road. These are existing model telemetry, not generated fixtures.
        call = {"initial_latents_sha256": "0224a69be29f7d347a134912222886b09d9b7655f566032a0f3baed09062e443"}
        result = {"actual_union_calls": {"2": {
            "encoded_control_sha256": "22cc2c110ec1a7cc9b599c0e0df8fff9cabfbcc931ccff5cd331be4dfa087ce7"}}}
        runtime = {"packages": {"diffusers": "0.41.0"}}
        args = [call, copy.deepcopy(call), result, copy.deepcopy(result), runtime, copy.deepcopy(runtime)]
        self.assertTrue(all(probe.pair_checks(*args).values()))
        for index, key in ((1, "same_initial_latents"), (3, "same_depth_encoding"), (5, "same_runtime")):
            changed = copy.deepcopy(args)
            if index == 1:
                changed[index]["initial_latents_sha256"] = "0" * 64
            elif index == 3:
                changed[index]["actual_union_calls"]["2"]["encoded_control_sha256"] = "0" * 64
            else:
                changed[index]["packages"]["diffusers"] = "changed"
            checks = probe.pair_checks(*changed)
            self.assertFalse(checks[key])
            self.assertEqual(sum(checks.values()), 2)

    def test_missing_edge_branch_is_refused(self):
        observed = {"2": {"calls": 24}, "0": {"calls": 24}}
        probe.core.check_control_visits(observed, [2, 0], 24)
        for bad in ({"2": {"calls": 24}}, {"2": {"calls": 24}, "0": {"calls": 1}}):
            with self.assertRaises(RuntimeError):
                probe.core.check_control_visits(bad, [2, 0], 24)

    def test_runtime_selection_preserves_supported_environment(self):
        cfg = probe.load_config(probe.DEFAULT_CONFIG)
        original = copy.deepcopy(cfg)
        for version in ("0.40.0", "0.41.0"):
            resolved = probe.resolved_config(cfg, {"packages": {"diffusers": version}})
            self.assertEqual(resolved["diffusers_version"], version)
        self.assertEqual(cfg, original)
        with self.assertRaises(RuntimeError):
            probe.resolved_config(cfg, {"packages": {"diffusers": "0.42.0"}})

    def test_code_hash_ignores_git_line_endings_but_not_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "code.py"
            path.write_bytes(b"x = 1\ny = 2\n")
            initial = probe.text_hash(path)
            path.write_bytes(b"x = 1\r\ny = 2\r\n")
            self.assertEqual(probe.text_hash(path), initial)
            path.write_bytes(b"x = 2\ny = 2\n")
            self.assertNotEqual(probe.text_hash(path), initial)


if __name__ == "__main__":
    unittest.main()
