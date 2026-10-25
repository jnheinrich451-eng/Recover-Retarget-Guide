"""Known-fixture contract tests; no mocked vision models or predicted labels."""
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

import yaml
from PIL import Image
from scripts import probe_m2d_proxy_style_selftest as probe


def answers(full, crop):
    return {v: {"observation": None if label is None else {"appearance": label, "evidence": "Decision fixture only"}}
            for v, label in (("full", full), ("subject_crop", crop))}


class FixtureContracts(unittest.TestCase):
    def test_correct_known_classes(self):
        self.assertTrue(probe.fixture_result("photographic", answers("photographic", "photographic"))["passed"])
        for label in ("cartoon", "stylized_render", "toy"):
            self.assertTrue(probe.fixture_result("nonphotographic", answers(label, label))["passed"])

    def test_constant_cartoon_classifier_cannot_pass_photo(self):
        out = probe.fixture_result("photographic", answers("cartoon", "cartoon"))
        self.assertFalse(out["passed"])

    def test_constant_photo_classifier_cannot_pass_cartoon(self):
        self.assertFalse(probe.fixture_result("nonphotographic", answers("photographic", "photographic"))["passed"])

    def test_unknown_missing_or_conflicting_views_fail(self):
        for a, b in ((None, None), ("uncertain", "photographic"), ("photographic", "cartoon"), ("cartoon", "toy")):
            for expected in ("photographic", "nonphotographic"):
                self.assertFalse(probe.fixture_result(expected, answers(a, b))["passed"])

    def test_bad_expected_class_refused(self):
        with self.assertRaisesRegex(RuntimeError, "Unknown fixture"):
            probe.fixture_result("anything", answers("cartoon", "cartoon"))


class ActualFixtureInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = probe.ROOT / "measurements/m2d"
        cls.photos = cls.root / "035_bear_style_selftest_inputs_v1"
        cls.source = cls.root / "035_bear_style_filter_v1"
        if not (cls.photos / "usfs_brown.jpg").exists() or not (cls.source / "index.json").exists():
            raise unittest.SkipTest("Actual photographs / returned GPU evidence absent")

    def test_returned_all_cartoon_is_actual_inference_not_parser_default(self):
        cfg = yaml.safe_load(probe.style.DEFAULT.read_text(encoding="utf-8"))
        probe.style.facing.resume.verify_files(self.source, probe.core.read_json(self.source / "index.json"))
        count = 0
        for seed in (4, 5, 6, 7):
            observed = probe.style.answers_for(self.source, seed, cfg)
            for answer in observed.values():
                self.assertEqual(answer["observation"]["appearance"], "cartoon")
                self.assertEqual(len(answer["attempts"]), 1)
                self.assertIsNone(answer["attempts"][0]["parse_error"])
                count += 1
        self.assertEqual(count, 8)

    def test_prepare_actual_photos_and_cartoon_preserves_bytes_and_resumes(self):
        with tempfile.TemporaryDirectory() as temp:
            args = Namespace(config=probe.DEFAULT, probe_root=self.root, output_dir=Path(temp) / "selftest",
                             photo_dir=self.photos, device="cpu", gpu_memory_gib=None)
            spec, cfg = probe.prepare(args)
            probe.prepare(args)
            for f in spec["fixtures"]:
                folder = args.output_dir / f["id"]
                if "url" in f:
                    self.assertEqual(probe.core.file_hash(folder / "source.jpg"), f["sha256"])
                    with Image.open(folder / "full.png") as pixels:
                        self.assertEqual(list(pixels.size), f["source_size_wh"])
                else:
                    self.assertEqual(probe.core.file_hash(folder / "full.png"), f["sha256"])
            self.assertFalse((args.output_dir / "results.json").exists())
            self.assertEqual(cfg["prompt"], probe.core.read_json(self.source / "manifest.json")["config"]["prompt"])

    def test_bad_photo_bytes_are_refused(self):
        spec = yaml.safe_load(probe.DEFAULT.read_text(encoding="utf-8"))
        f = spec["fixtures"][0]
        with self.assertRaisesRegex(RuntimeError, "Photo bytes changed"):
            probe.validate_blob(b"invalid or replaced download", f)


if __name__ == "__main__":
    unittest.main()
