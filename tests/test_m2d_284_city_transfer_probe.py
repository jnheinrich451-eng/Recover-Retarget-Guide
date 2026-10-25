"""Contract tests for the isolated city comparison; no mocked model inference."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts import probe_m2d_284_city_transfer as probe


class CityTransferProbeTests(unittest.TestCase):
    def test_prompt_changes_only_environment(self):
        cfg = probe.load_config(probe.DEFAULT_CONFIG)
        record = probe.resume.read(probe.ROOT / cfg["cases"][0]["m2d_generation_record"])
        original = copy.deepcopy(record)
        prompt = probe.city_prompt(record, cfg["prompt_replacement"])
        self.assertEqual(prompt, "a red car driving, on a paved city street, rear three-quarter view, "
                                 "moving away from the camera, medium-wide shot")
        self.assertEqual(record, original)
        self.assertEqual(list(probe.hybrid.arms(cfg)), ["plain", "e000", "e040"])

    def test_stale_or_ambiguous_prompt_refused(self):
        cfg = probe.load_config(probe.DEFAULT_CONFIG)
        record = probe.resume.read(probe.ROOT / cfg["cases"][0]["m2d_generation_record"])
        for bad in ("on a forest track", "car, on a forest track, on a forest track, away", "car, on grass, away"):
            changed = copy.deepcopy(record)
            changed["prompt"]["prompt"] = bad
            with self.assertRaises(RuntimeError):
                probe.city_prompt(changed, cfg["prompt_replacement"])

    def test_identity_failure_is_persisted_and_stops(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for seed, differs in ((0, False), (1, True)):
                root = output / "generation/284_vehicle" / f"seed_{seed}"
                for arm in ("plain", "e000"):
                    out = root / arm
                    probe.core.save_image(out / "image.png", Image.new("RGB", (4, 4),
                                          "white" if differs and arm == "e000" else "black"))
                    probe.core.save_json(out / "result.json", {"image_sha256": probe.core.file_hash(out / "image.png")})
                if differs:
                    with self.assertRaises(RuntimeError):
                        probe.identity_check(output, seed)
                    self.assertFalse(probe.resume.read(root / "identity.json")["hook_path_identical"])
                else:
                    self.assertTrue(probe.identity_check(output, seed)["hook_path_identical"])

    def test_completed_refuses_changed_prompt_or_missing_union_execution(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            expected = {"prompt": "city", "edge_scale": 0.4}
            probe.core.save_json(out / "call.json", expected)
            probe.core.save_image(out / "image.png", Image.new("RGB", (4, 4)))
            result = {"image_sha256": probe.core.file_hash(out / "image.png"), "transformer_calls": 30, "union_calls": 30}
            probe.core.save_json(out / "result.json", result)
            self.assertTrue(probe.completed(out, expected, 30))
            with self.assertRaises(RuntimeError):
                probe.completed(out, dict(expected, prompt="forest"), 30)
            result["union_calls"] = 0
            (out / "result.json").write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                probe.completed(out, expected, 30)


if __name__ == "__main__":
    unittest.main()
