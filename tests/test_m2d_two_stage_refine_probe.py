"""CPU tests of the two-stage refine probe's arithmetic, instruments and isolation guard (no FLUX, no mocks)."""
import unittest
from pathlib import Path

import numpy as np
import yaml

from scripts import probe_m2d_two_stage_refine as probe


def diffusers_executed(steps, strength):
    """FluxControlImg2ImgPipeline.get_timesteps, as shipped in Diffusers 0.39-0.41."""
    init_timestep = min(steps * strength, steps)
    return steps - int(max(steps - init_timestep, 0))


class TwoStageRefineTests(unittest.TestCase):
    def test_config_matches_production_sampling(self):
        cfg = probe.load_config(probe.DEFAULT_CONFIG)
        self.assertEqual((cfg["steps"], cfg["guidance"]), (30, 10.0))
        self.assertEqual(sum(len(c["seeds"]) for c in cfg["cases"]) * len(cfg["refine_steps"]), 36)

    def test_strength_realises_exact_step_counts(self):
        for k in range(1, 30):
            self.assertEqual(diffusers_executed(30, probe.strength_for(k, 30)), k)

    def test_instruments_on_known_answers(self):
        rng = np.random.default_rng(0)
        texture = (rng.random((256, 256)) * 255).astype(np.float32)
        smooth = np.tile(np.linspace(0, 255, 256, dtype=np.float32), (256, 1))
        report = probe.self_test(texture + smooth, 16)
        self.assertTrue(report["detail_monotone"] and report["layout_monotone"])
        with self.assertRaises(RuntimeError):
            probe.self_test(np.full((256, 256), 128, np.float32), 16)   # constant image: no detail to lose

    def test_output_guard_refuses_overlap_and_frozen_records(self):
        source = Path("/content/drive/MyDrive/motion_transfer/probe/m2d/road_generalization_v1/generation/370_human")
        for bad in (source, source / "refine", source.parent,
                    probe.ROOT / "measurements/m2e/refine", Path("/content/drive/MyDrive/motion_transfer/cache/m2_d/x")):
            with self.assertRaises(RuntimeError):
                probe.guard_output(bad.resolve(), [source.resolve()])
        probe.guard_output(Path("/content/drive/MyDrive/motion_transfer/probe/m2d/two_stage_refine_v1").resolve(),
                           [source.resolve()])

    def test_cases_reference_tracked_frozen_records(self):
        cfg = yaml.safe_load(probe.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        for case in cfg["cases"]:
            self.assertTrue((probe.ROOT / case["m2d_generation_record"]).is_file())
            if case["frozen_anchor"]:
                self.assertTrue((probe.ROOT / case["frozen_anchor"]).is_file())


if __name__ == "__main__":
    unittest.main()
