"""CPU-only integrity tests; no estimator or reconstructor is mocked or invoked."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from scripts import probe_bear_retarget_geometry as p


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_numpy_metadata_and_immutable_write(self):
        path = self.root / "data.json"
        value = {"K": np.eye(3), "n": np.int64(4), "invalid": np.float64(np.nan)}
        p.save_json(path, value)
        p.save_json(path, value)
        self.assertEqual(p.read(path), {"K": np.eye(3).tolist(), "n": 4, "invalid": None})
        with self.assertRaisesRegex(RuntimeError, "Changed artifact"):
            p.save_json(path, {"n": 5})

    def test_resume_checks_all_saved_bytes(self):
        folder, complete = p.stage(self.root, "depth")
        self.assertFalse(complete)
        p.save_npz(folder / "data.npz", data=np.arange(6).reshape(2, 3))
        p.finish(folder, status="ready")
        self.assertTrue(p.stage(self.root, "depth")[1])
        with (folder / "data.npz").open("ab") as f:
            f.write(b"corruption")
        with self.assertRaisesRegex(RuntimeError, "Changed checkpoint"):
            p.stage(self.root, "depth")

    def test_partial_attempt_is_preserved_before_retry(self):
        folder, _ = p.stage(self.root, "reconstruction/seeds/seed_0")
        p.put(folder / "failure.txt", b"interrupted attempt")
        retried, complete = p.stage(self.root, "reconstruction/seeds/seed_0")
        self.assertFalse(complete)
        self.assertEqual(retried, folder)
        self.assertEqual(list(folder.iterdir()), [])
        archived = list((self.root / "partial_attempts").rglob("failure.txt"))
        self.assertEqual(len(archived), 1)
        self.assertEqual(archived[0].read_bytes(), b"interrupted attempt")

    def test_output_cannot_overwrite_cache_or_evidence(self):
        cfg = {"inputs": {"anchor": {"path": "old/anchor.png"}}}
        args = SimpleNamespace(drive_root=self.root, probe_root=self.root / "probe",
                               output_dir=self.root / "probe/new")
        p.validate_output(args, cfg)
        for output in (args.probe_root, args.probe_root / "old", args.probe_root / "old/anchor.png"):
            args.output_dir = output
            with self.assertRaises(RuntimeError):
                p.validate_output(args, cfg)
        args.probe_root = self.root / "cache"
        args.output_dir = args.probe_root / "new"
        with self.assertRaisesRegex(RuntimeError, "protected directory"):
            p.validate_output(args, cfg)


class ReturnedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = p.yaml.safe_load(p.DEFAULT_CONFIG.read_text())
        args = SimpleNamespace(probe_root=p.ROOT / "measurements/m2d")
        try:
            cls.inputs = {k: p.resolve_input(args, v) for k, v in cls.cfg["inputs"].items()}
        except FileNotFoundError as exc:
            raise unittest.SkipTest(f"Returned GPU artifacts are not in this checkout: {exc}")

    def test_final_image_mask_and_camera_share_provenance(self):
        src = self.inputs
        rec, placement = p.read(src["mask_record"]), p.read(src["placement"])
        self.assertEqual(rec["inputs"]["image_sha256"], p.digest(src["anchor"]))
        self.assertEqual(rec["files"]["subject.png"], p.digest(src["anchor_mask"]))
        self.assertEqual(placement["clip"], self.cfg["clip"])
        self.assertEqual(placement["t_a"], self.cfg["source_frame"])
        self.assertEqual(placement["category"], "bear")
        K = np.diag([4., 4., 1.]) @ np.asarray(placement["stage0"]["K_a"])
        np.testing.assert_allclose(K, rec["K_analysis"])
        am = p.mask(src["anchor_mask"])
        control = p.arrays(src["control"])
        self.assertEqual(am.shape, (1024, 1024))
        self.assertTrue((am & control["M_obj"]).any())
        for key in ("M_obj", "valid", "zeta_ctrl", "zeta_obj"):
            self.assertEqual(control[key].shape, am.shape)

    def test_real_proxy_renders_without_mutating_vertices(self):
        # R1 mesh exercises the existing renderer API, not a claimed R2 result.
        path = self.inputs["proxy_qc"].with_name("proxy.npz")
        if not path.exists():
            self.skipTest("R1 mesh not copied to this checkout")
        mesh = p.arrays(path)
        vertices = mesh["vertices"] - mesh["vertices"].mean(0)
        original = vertices.copy()
        depth, mask, cell_size = p.renderer.render_orthographic(vertices, mesh["faces"], 64)
        self.assertEqual(depth.shape, (64, 64))
        self.assertTrue(mask.any())
        self.assertTrue(np.isfinite(depth[mask]).all())
        self.assertGreater(cell_size, 0)
        np.testing.assert_array_equal(vertices, original)


if __name__ == "__main__":
    unittest.main()
