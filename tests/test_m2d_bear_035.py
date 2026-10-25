"""Input/provenance/geometry boundary checks; no external model mocks."""
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from scripts import probe_m2d_bear_035 as probe
from scripts import m2d_bear_reconstruct as reconstruct


class BearProbeTests(unittest.TestCase):
    def test_sam2_distribution_name_with_real_metadata_reader(self):
        # Packaging fixture only: no substitute SAM2 module or model inference.
        with tempfile.TemporaryDirectory() as tmp:
            metadata = Path(tmp) / "sam_2-1.0.dist-info/METADATA"
            metadata.parent.mkdir()
            metadata.write_text("Metadata-Version: 2.1\nName: SAM-2\nVersion: 1.0\n", encoding="utf-8")
            sys.path.insert(0, tmp)
            try:
                with self.assertRaises(PackageNotFoundError):
                    version("sam2")  # Reproduce the reported failure.
                self.assertEqual(probe.package_versions(probe.MASK_PACKAGES[-1:]), {"SAM-2": "1.0"})
                with self.assertRaises(PackageNotFoundError):
                    probe.package_versions(("nonexistent-bear-probe-distribution",))
            finally:
                sys.path.remove(tmp)

    def metadata_failure(self, out):
        cfg, _, _ = probe.configuration()
        code = probe.code_hashes(cfg)
        code["scripts/probe_m2d_bear_035.py"] = probe.PRE_METADATA_FIX_DRIVER
        probe.save_json(out / "experiment.json", {"config": cfg, "code": code})
        probe.core.immutable_bytes(out / "inputs/reference.png", b"input artifact for provenance test")
        record = {"config": cfg, "code": code,
                  "files": {"inputs/reference.png": probe.core.file_hash(out / "inputs/reference.png")}}
        probe.save_json(out / "manifest.json", record)
        for seed in cfg["proxy_seeds"]:
            folder = out / "proxies" / f"seed_{seed}"
            probe.core.immutable_bytes(folder / "image.png", f"artifact bytes {seed}".encode())
            probe.finish(folder, ["image.png"])
        return cfg, record

    def test_metadata_fix_preserves_generated_artifacts_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cfg, record = self.metadata_failure(out)
            old_bytes = {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}
            self.assertEqual(probe.verified(out, cfg), record)
            repair = out / "provenance/sam2_metadata_fix.json"
            self.assertEqual(probe.core.read_json(repair)["code"], probe.code_hashes(cfg))
            self.assertEqual(probe.core.read_json(repair)["original_manifest_sha256"],
                             probe.core.file_hash(out / "manifest.json"))
            # Subsequent stages can resume after the exact repair is registered.
            probe.save_json(out / "masks/seed_0/result.json", {"progress": "after repair"})
            self.assertEqual(probe.verified(out, cfg), record)
            for path, data in old_bytes.items():
                self.assertEqual((out / path).read_bytes(), data)
            changed = probe.core.read_json(repair)
            changed["original_manifest_sha256"] = "tampered"
            repair.write_text(json.dumps(changed))
            with self.assertRaisesRegex(RuntimeError, "Existing artifact differs"):
                probe.verified(out, cfg)

    def test_metadata_fix_refuses_other_changes_and_later_work(self):
        for change in ("calibration", "driver", "config", "input", "proxy", "masks", "controls", "runtime"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                cfg, record = self.metadata_failure(out)
                if change == "calibration":
                    record["code"]["configs/calibration.lock.yaml"] = "another version"
                elif change == "driver":
                    record["code"]["scripts/probe_m2d_bear_035.py"] = "unknown version"
                elif change == "config":
                    record["config"] = {**cfg, "target_phrase": "different prompt"}
                elif change in ("input", "proxy"):
                    relative = "inputs/reference.png" if change == "input" else "proxies/seed_0/image.png"
                    (out / relative).write_bytes(b"changed")
                else:
                    relative = "runtimes/masks.json" if change == "runtime" else f"{change}/result.json"
                    probe.save_json(out / relative, {})
                (out / "manifest.json").write_text(json.dumps(record))
                with self.assertRaises(RuntimeError):
                    probe.verified(out, cfg)
                self.assertFalse((out / "provenance/sam2_metadata_fix.json").exists())

    def test_placement_reader_record_can_be_saved_without_mutation(self):
        from src.anchor.m2c import placement
        original = {"schema": "m2c_placement_v1", "admitted": True, "proxy_cache_key": "cpu-fixture",
                    "proxy_facing": "right", "mode": "depth_preserving",
                    "placement": {"X": [1., 2., 3.], "R_A": np.eye(3).tolist(), "rho_tilde": [0., 0., 1.]}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probe.core.save_json(root / "placement.json", original)
            record = placement.read_placement(root)  # actual reader caused the reported failure
            self.assertIsInstance(record["placement"]["R_A"], np.ndarray)
            with self.assertRaises(TypeError):
                json.dumps(record)
            probe.save_json(root / "saved.json", record)
            self.assertEqual(probe.core.read_json(root / "saved.json"), original)
            self.assertIsInstance(record["placement"]["R_A"], np.ndarray)
            probe.save_json(root / "saved.json", record)  # identical retry

    def test_actual_c2_record_numpy_boolean_serializes(self):
        from src.anchor.m2c import sanitise
        import yaml
        params = yaml.safe_load((probe.ROOT / "configs/m2c.yaml").read_text())["m2c"]["c2"]
        y, x = np.mgrid[:256, :256]
        depth = 1. / (1. + .001 * x + .002 * y)
        mask = np.zeros((256, 256), bool)
        mask[100:150, 100:150] = True
        _, _, record = sanitise.sanitise(depth, mask, np.array([[200, 0, 128], [0, 200, 128], [0, 0, 1.]]), params)
        self.assertTrue(record["admitted"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c2.json"
            probe.save_json(path, record)
            result = probe.core.read_json(path)
            self.assertIs(result["gates"]["g_ring_valid"]["collinear"], False)

    def test_numpy_values_stay_numeric_and_unknown_types_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "record.json"
            data = {"nested": [np.array([1, 2]), {"flag": np.bool_(True), "count": np.int64(3), "value": np.float32(.5)}]}
            probe.save_json(path, data)
            self.assertEqual(probe.core.read_json(path), {"nested": [[1, 2], {"flag": True, "count": 3, "value": .5}]})
            with self.assertRaises(RuntimeError):
                probe.save_json(path, {"changed": True})
            with self.assertRaises(TypeError):
                probe.save_json(Path(tmp) / "bad.json", {"unsupported": object()})

    def legacy_prepare(self, out):
        cfg, _, _ = probe.configuration()
        code = probe.code_hashes(cfg)
        code.update(probe.PRE_JSON_FIX_CODE)
        record = {"config": cfg, "code": code}
        probe.core.save_json(out / "experiment.json", record)
        probe.core.immutable_bytes(out / "inputs/reference.png", b"partial reference bytes")
        return cfg, record

    def test_known_failed_prepare_resumes_with_original_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cfg, old = self.legacy_prepare(out)
            old_bytes = (out / "experiment.json").read_bytes()
            probe.register_preparation(out, cfg)
            self.assertEqual(probe.core.read_json(out / "experiment.json")["code"], probe.code_hashes(cfg))
            archived = list((out / "partial_attempts/prepare_json_fix").iterdir())
            self.assertEqual(len(archived), 1)
            self.assertEqual((archived[0] / "experiment.json").read_bytes(), old_bytes)
            self.assertEqual((archived[0] / "inputs/reference.png").read_bytes(), b"partial reference bytes")
            self.assertFalse((out / "inputs").exists())
            probe.register_preparation(out, cfg)
            self.assertEqual(len(list((out / "partial_attempts/prepare_json_fix").iterdir())), 1)

    def test_prepare_recovery_refuses_completed_or_generated_work(self):
        for name in ("manifest.json", "proxies/seed_0/image.png", "sam3d_seeds/seed_0/mesh.npz", "anchors/seed_0/bear/image.png"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                cfg, _ = self.legacy_prepare(out)
                probe.core.immutable_bytes(out / name, b"existing output")
                with self.assertRaises(RuntimeError):
                    probe.register_preparation(out, cfg)
                self.assertFalse((out / "partial_attempts").exists())

    def test_prepare_recovery_refuses_unrelated_code_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cfg, old = self.legacy_prepare(out)
            old["code"]["src/anchor/m2c/placement.py"] = "unknown revision"
            (out / "experiment.json").write_text(json.dumps(old))
            with self.assertRaisesRegex(RuntimeError, "Unknown prior code"):
                probe.register_preparation(out, cfg)
            self.assertTrue((out / "inputs/reference.png").exists())

    def test_only_subject_phrase_changes(self):
        cfg, settings, prompts = probe.configuration()
        self.assertEqual(prompts["vehicle"][len(cfg["source_phrase"]):], prompts["bear"][len(cfg["target_phrase"]):])
        self.assertIn("cliff road", prompts["bear"])
        self.assertNotIn("turn", prompts["bear"])
        self.assertEqual(cfg["active_steps"], [12, 12])
        self.assertEqual((settings["depth_scale"], settings["edge_scale"]), (.6, .2))

    def test_trace_has_boundaries_no_lane_markings(self):
        cfg, _, _ = probe.configuration()
        trace = probe.core.read_json(probe.ROOT / cfg["road_trace"])
        self.assertEqual(trace["source_frame"], 4)
        self.assertTrue(all("boundary" in p["name"] for p in trace["polylines"]))
        self.assertFalse(any("marking" in p["name"] for p in trace["polylines"]))
        # Verify the actual archived reference and unchanged control-grid shape.
        self.assertEqual(probe.core.reference_image(trace).size, (1024, 1024))

    def test_road_is_hidden_by_new_subject_only(self):
        road = np.zeros((32, 32), np.uint8)
        road[16, :] = 255
        old_car = np.zeros_like(road, bool)
        old_car[10:22, 10:24] = True
        bear = np.zeros_like(old_car)
        bear[8:24, 13:18] = True
        edges = np.zeros_like(road)
        visible_road, combined = probe.core.compose_edges(road, edges, bear)
        self.assertFalse(visible_road[bear].any())
        self.assertEqual(visible_road[16, 11], 255)  # road revealed outside narrower bear
        self.assertTrue(old_car[16, 11])
        self.assertTrue(np.array_equal(combined[bear], edges[bear]))

    def test_no_background_mask_fallback(self):
        cfg, _, _ = probe.configuration()
        masks = np.ones((3, 32, 32), bool)
        self.assertIsNone(probe.select_mask(masks, [1, 1, 1], cfg["segmentation"]))
        masks[1] = False
        masks[1, 10:20, 4:28] = True
        self.assertEqual(probe.select_mask(masks, [1, .5, 1], cfg["segmentation"]), 1)

    def test_frozen_paths_are_refused(self):
        drive = Path("/test-drive")
        for path in (drive, drive / "cache", drive / "cache/m2_b2/bear", probe.ROOT / "src/new", probe.ROOT / "measurements/m2e"):
            with self.assertRaises(RuntimeError):
                probe.guard_output(path, drive)
        probe.guard_output(drive / "probe/m2d/035_bear_v1", drive)

    def test_completed_artifacts_are_checked_not_just_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            probe.core.immutable_bytes(p / "image.png", b"artifact bytes")
            probe.finish(p, ["image.png"])
            self.assertTrue(probe.finished(p))
            (p / "image.png").write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                probe.finished(p)

    def test_incomplete_attempt_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "proxies/seed_0"
            probe.core.save_json(folder / "call.json", {"seed": 0})
            archived = probe.resume.archive_incomplete(folder, root)
            self.assertFalse(folder.exists())
            self.assertEqual(json.loads((archived / "call.json").read_text()), {"seed": 0})

    def test_actual_facing_required_and_pick_is_bound_to_pixels(self):
        cfg, _, _ = probe.configuration()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "observed"):
                reconstruct.approval_record(out, 0, None, cfg)
            image = out / "proxies/seed_0"
            mask = out / "masks/seed_0"
            probe.core.save_image(image / "image.png", np.zeros((32, 32, 3), np.uint8))
            probe.finish(image, ["image.png"])
            probe.core.save_image(mask / "mask.png", np.ones((32, 32), np.uint8) * 255)
            probe.finish(mask, ["mask.png"], status="needs_review", image_sha256=probe.core.file_hash(image / "image.png"))
            left = reconstruct.approval_record(out, 0, "left", cfg)
            right = reconstruct.approval_record(out, 0, "right", cfg)
            self.assertNotEqual(probe.resume.fingerprint(left), probe.resume.fingerprint(right))
            probe.core.save_json(out / "proxy_approval.json", left)
            with self.assertRaises(RuntimeError):
                probe.core.save_json(out / "proxy_approval.json", right)

    def test_refused_mask_cannot_be_reconstructed(self):
        cfg, _, _ = probe.configuration()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            image, mask = out / "proxies/seed_0", out / "masks/seed_0"
            probe.core.save_image(image / "image.png", np.zeros((32, 32, 3), np.uint8))
            probe.finish(image, ["image.png"])
            probe.finish(mask, [], status="refused", reason="no detection")
            with self.assertRaisesRegex(RuntimeError, "refused"):
                reconstruct.approval_record(out, 0, "right", cfg)

    def test_geometry_previews_on_known_box(self):
        # Real CPU renderer, analytic geometry; not a model substitution.
        V = np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                      [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]], float)
        F = np.array([[0,1,2], [0,2,3], [4,5,6], [4,6,7], [0,4,7], [0,7,3],
                      [1,5,6], [1,6,2], [0,1,5], [0,5,4], [3,2,6], [3,6,7]])
        with tempfile.TemporaryDirectory() as tmp:
            names = reconstruct.mesh_previews(Path(tmp), V, F)
            self.assertEqual(len(names), 3)
            for name in names:
                self.assertTrue((Path(tmp) / name).exists())


if __name__ == "__main__":
    unittest.main()
