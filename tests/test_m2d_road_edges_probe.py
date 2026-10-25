"""CPU geometry and evidence checks; no mock external model or generated output."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("road_probe", ROOT / "scripts/probe_m2d_road_edges.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class RoadProbeTests(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load(probe.DEFAULT_CONFIG.read_text())
        self.trace = probe.read_json(probe.DEFAULT_TRACE)

    def test_archived_reference_coordinates_and_prompt(self):
        reference = probe.reference_image(self.trace)
        self.assertEqual(reference.size, (1024, 1024))
        handoff = probe.read_json(ROOT / self.trace["handoff"])
        self.assertEqual(handoff["t_a"], self.trace["source_frame"])
        baseline = probe.read_json(ROOT / self.trace["baseline_record"])
        previous = probe.read_json(ROOT / "measurements/m2d_road_prompt_probe/experiment.json")
        expected = baseline["prompt"]["prompt"].replace(baseline["target"]["scene"], previous["scene"])
        self.assertEqual(self.cfg["prompt"], expected)

    def test_open_lines_do_not_close_on_image_border(self):
        trace = {"size": [32, 32], "polylines": [
            {"evidence": "visible", "points": [[0, 5], [15, 5], [31, 20]]},
            {"evidence": "inferred", "points": [[10, 25], [20, 25]]}]}
        visible = probe.draw_road(trace, 1, include_inferred=False)
        all_edges = probe.draw_road(trace, 1)
        self.assertEqual(int(visible[5, 10]), 255)
        self.assertEqual(int(visible[25, 15]), 0)
        self.assertEqual(int(all_edges[25, 15]), 255)
        self.assertFalse(visible[31].any())
        self.assertFalse(visible[0].any())
        self.assertEqual(int(visible[10, 10]), 0)  # would lie on a closing diagonal

    def test_target_occlusion_and_road_ablation(self):
        # Analytic sloping rectangle with a foreground occluder. Tests geometry
        # operations, not a simulated reconstructor or a mock generative model.
        n = 64
        cfg = dict(self.cfg, size=n)
        mask = np.zeros((n, n), bool)
        mask[12:52, 16:48] = True
        z = np.zeros((n, n), np.float32)
        z[mask] = np.broadcast_to(np.linspace(0.2, 0.8, n, dtype=np.float32), (n, n))[mask]
        combined = z.copy()
        combined[12:20, 16:48] = 1
        visible, car = probe.target_controls({"D_ctrl": combined, "zeta_ctrl": combined,
                                               "zeta_obj": z, "M_obj": mask}, cfg)
        self.assertFalse(visible[12:20].any())
        self.assertTrue(visible[20:52, 16:48].all())
        self.assertFalse(car[~visible].any())
        road = np.zeros((n, n), np.uint8)
        road[32, :] = 255
        clipped, full = probe.compose_edges(road, car, visible)
        self.assertFalse(clipped[visible].any())
        np.testing.assert_array_equal(full[visible], car[visible])
        np.testing.assert_array_equal(full[~visible], clipped[~visible])
        self.assertEqual(int(full[32, 0]), 255)
        self.assertFalse(np.array_equal(full, car))

    def test_nonfinite_depth_is_refused(self):
        n = 8
        d = np.ones((n, n), np.float32)
        d[0, 0] = np.nan
        with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
            probe.target_controls({"D_ctrl": d, "zeta_ctrl": d, "zeta_obj": d,
                                   "M_obj": np.ones((n, n), bool)}, dict(self.cfg, size=n))

    def test_resume_rejects_changed_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            probe.save_json(path, {"seed": 0})
            first = path.read_bytes()
            probe.save_json(path, {"seed": 0})
            with self.assertRaisesRegex(RuntimeError, "Existing artifact differs"):
                probe.save_json(path, {"seed": 1})
            self.assertEqual(path.read_bytes(), first)

    def test_real_geometry_fixtures_complement(self):
        from src.anchor.m2d.stimuli import s1_ground_box, s2_crossing_occluder, s3_corridor
        for factory in (s1_ground_box, s2_crossing_occluder, s3_corridor):
            fixture = factory()
            self.assertEqual(fixture["D_ctrl"].shape, (1024, 1024))
            self.assertTrue(np.isfinite(fixture["D_ctrl"]).all())
            np.testing.assert_allclose(fixture["D_ctrl"] + fixture["D_ctrl_inv"], 1, atol=0, rtol=0)

    def test_missing_or_truncated_edge_conditioning_is_refused(self):
        # Validate telemetry, not a simulated model. A missing second branch is
        # otherwise silently ignored by zip() in the Union control implementation.
        probe.check_control_visits({"2": {"calls": 24}, "0": {"calls": 24}}, [2, 0], 24)
        for record in ({"2": {"calls": 24}}, {"2": {"calls": 24}, "0": {"calls": 1}}):
            with self.assertRaisesRegex(RuntimeError, "not executed"):
                probe.check_control_visits(record, [2, 0], 24)


if __name__ == "__main__":
    unittest.main()
