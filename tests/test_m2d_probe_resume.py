"""Provenance/checkpoint tests; no external model is invoked or mocked."""
import copy
import tempfile
from pathlib import Path
import unittest

from scripts import m2d_probe_resume as resume


class ResumeTests(unittest.TestCase):
    def manifest(self):
        return {"schema": "road_generalization_probe_v1", "suite": {"size": 1024},
                "source_sha256": {resume.DRIVER: "current", resume.HELPER: "helper", "control.py": "fixed"},
                "packages": {"numpy": "new"}, "status": "probe"}

    def test_known_migration_preserves_original_and_refuses_other_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            current = self.manifest()
            old = copy.deepcopy(current)
            old["source_sha256"][resume.DRIVER] = resume.LEGACY_DRIVER
            del old["source_sha256"][resume.HELPER]
            old["packages"]["numpy"] = "old"
            resume.core.save_json(root / "suite_manifest.json", old)
            original = (root / "suite_manifest.json").read_bytes()
            ref = resume.register_suite(root, current)
            self.assertEqual(ref, resume.register_suite(root, current))
            self.assertEqual(original, (root / "suite_manifest.json").read_bytes())
            resume.verify_reference(ref, root)
            for section, name in (("suite", "size"), ("source_sha256", "control.py")):
                changed = copy.deepcopy(current)
                changed[section][name] = "changed"
                with self.assertRaises(RuntimeError):
                    resume.register_suite(root, changed)

    def test_unknown_driver_migration_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            current = self.manifest()
            old = copy.deepcopy(current)
            old["source_sha256"][resume.DRIVER] = "unknown"
            resume.core.save_json(Path(temp) / "suite_manifest.json", old)
            with self.assertRaises(RuntimeError):
                resume.register_suite(temp, current)

    def test_runtime_transition_and_legacy_attribution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            suite_ref = resume.register_suite(root, self.manifest())
            old_ref = resume.register_runtime(root, "detector", {"device": "old"})
            old_bytes = (root / "detector/runtime.json").read_bytes()
            new_ref = resume.register_runtime(root, "detector", {"device": "new"})
            self.assertNotEqual(old_ref, new_ref)
            self.assertEqual(old_bytes, (root / "detector/runtime.json").read_bytes())
            legacy = root / "detector/clip/frame_0"
            resume.record_output_provenance(legacy, root, "detector")
            self.assertEqual(resume.read(legacy / "provenance.json")["runtime"]["sha256"],
                             resume.core.file_hash(root / "detector/runtime.json"))
            fresh = root / "detector/clip/frame_1"
            session = {"runtime": new_ref, "suite": suite_ref, "origin": "test metadata"}
            resume.record_output_provenance(fresh, root, "detector", session)
            resume.record_output_provenance(fresh, root, "detector")
            self.assertEqual(resume.read(fresh / "provenance.json"), session)

    def test_partial_preserved_and_completed_folder_not_moved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = root / "detector/clip/frame_0"
            out.mkdir(parents=True)
            (out / "input.json").write_text('{"frame": 0}')
            (out / "unfinished.partial").write_bytes(b"partial data")
            archive = resume.archive_incomplete(out, root)
            self.assertEqual((archive / "unfinished.partial").read_bytes(), b"partial data")
            self.assertFalse(out.exists())
            out.mkdir()
            (out / "result.json").write_text("completion marker")
            with self.assertRaises(RuntimeError):
                resume.archive_incomplete(out, root)
            with self.assertRaises(RuntimeError):
                resume.archive_incomplete(root.parent, root)

    def test_artifact_corruption_and_path_escape_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            path = out / "artifact.txt"
            path.write_text("original artifact")
            ledger = {path.name: resume.core.file_hash(path)}
            resume.verify_files(out, ledger)
            path.write_text("changed")
            with self.assertRaises(RuntimeError):
                resume.verify_files(out, ledger)
            with self.assertRaises(RuntimeError):
                resume.verify_files(out, {"../outside": "unused"})


if __name__ == "__main__":
    unittest.main()
