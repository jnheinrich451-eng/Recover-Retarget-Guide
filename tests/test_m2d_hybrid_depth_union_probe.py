"""CPU tests of the hybrid probe's config, arms, instruments and identity gate (no FLUX, no Union, no mocks)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import probe_m2d_hybrid_depth_union as probe


class HybridProbeTests(unittest.TestCase):
    def test_config_matches_production_and_arms(self):
        cfg = probe.load_config(probe.DEFAULT_CONFIG)
        self.assertEqual((cfg["steps"], cfg["guidance"]), (30, 10.0))
        names = probe.arms(cfg)
        self.assertEqual(list(names)[:2], ["plain", "e000"])
        self.assertIsNone(names["plain"])
        self.assertEqual(names["e000"], {"scale": 0.0, "union_steps": 30})
        self.assertEqual(sum(len(c["seeds"]) for c in cfg["cases"]) * len(names), 48)

    def test_v2_early_step_arms_match_diffusers_guidance_end(self):
        cfg = probe.load_config(probe.ROOT / "configs/m2d_hybrid_depth_union_probe_v2.yaml")
        names = probe.arms(cfg)
        self.assertEqual(list(names), ["plain", "e000", "e040_u30", "e040_u15", "e040_u09",
                                       "e070_u30", "e070_u15", "e070_u09"])
        self.assertEqual(names["e000"]["union_steps"], 30)
        self.assertEqual(sum(len(c["seeds"]) for c in cfg["cases"]) * len(names), 40)
        steps = cfg["steps"]
        for spec in filter(None, names.values()):
            end = spec["union_steps"] / steps
            # FluxControlNetPipeline: keep_i = 1 - (i / n < start or (i + 1) / n > end), start = 0
            kept = [i for i in range(steps) if not ((i + 1) / steps > end)]
            self.assertEqual(kept, list(range(spec["union_steps"])))
        self.assertEqual(probe.expected_call(cfg, "c", 0, "plain", None, "p", self._controls())["union_steps"], None)

    @staticmethod
    def _controls():
        from PIL import Image
        blank = Image.fromarray(np.zeros((4, 4, 3), np.uint8))
        return {"depth.png": blank, "road_edges.png": blank}

    def test_instruments_on_known_answers(self):
        rng = np.random.default_rng(0)
        g = (rng.random((256, 256)) * 128 + np.tile(np.linspace(0, 120, 256), (256, 1))).astype(np.float32)
        report = probe.self_test(g, 16)
        self.assertTrue(report["detail_contrast_invariant"] and report["detail_monotone"])
        with self.assertRaises(RuntimeError):
            probe.self_test(np.full((256, 256), 100, np.float32), 16)

    def test_identity_gate_stops_on_any_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed_dir = root / "generation/035_vehicle/seed_0"
            for arm, sha in (("plain", "a" * 64), ("e000", "a" * 64)):
                (seed_dir / arm).mkdir(parents=True)
                (seed_dir / arm / "result.json").write_text(json.dumps({"image_sha256": sha}))
            record = probe.identity_check(root, "035_vehicle", 0, "b" * 64)
            self.assertTrue(record["hook_path_identical"])
            self.assertFalse(record["plain_reproduces_frozen_anchor"])
            other = root / "other"
            seed_dir = other / "generation/035_vehicle/seed_0"
            for arm, sha in (("plain", "a" * 64), ("e000", "c" * 64)):
                (seed_dir / arm).mkdir(parents=True)
                (seed_dir / arm / "result.json").write_text(json.dumps({"image_sha256": sha}))
            with self.assertRaises(RuntimeError):
                probe.identity_check(other, "035_vehicle", 0, "a" * 64)

    def test_cases_reference_tracked_frozen_records(self):
        cfg = probe.load_config(probe.DEFAULT_CONFIG)
        for case in cfg["cases"]:
            record = json.loads((probe.ROOT / case["m2d_generation_record"]).read_text(encoding="utf-8"))
            self.assertEqual(record["pipeline"], "FluxControlPipeline")
            self.assertTrue(set(case["seeds"]) <= {c["seed"] for c in record["calls"]})


if __name__ == "__main__":
    unittest.main()
