"""Real saved controls/masks plus numerical policy checks; no model mocks."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts import probe_m2d_anchor_selection_crossclip as probe
from scripts import score_m2d_anchor_selection_crossclip as score


class ReturnedInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = probe.ROOT / "measurements/m2d"
        cls.cfg = probe.configuration()
        if not all((cls.root / cls.cfg[k] / "suite_manifest.json").exists() for k in ("bear_source", "road_source")):
            raise unittest.SkipTest("Returned inputs unavailable")
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name) / "crossclip"
        cls.m = probe.prepare(cls.cfg, cls.root, cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_prepare_is_idempotent_and_each_case_changes_only_seed(self):
        self.assertEqual(self.m, probe.prepare(self.cfg, self.root, self.output))
        self.assertEqual(self.m, probe.verified(self.output, self.cfg))
        for case in self.cfg["cases"]:
            calls = [probe.expected_call(self.output, self.m, case, s)[1] for s in self.cfg["seeds"]]
            for call in calls[1:]:
                self.assertEqual({k for k in call if call[k] != calls[0][k]}, {"seed"})
            self.assertEqual(calls[0]["scales"], [.6, .4])
            self.assertEqual(calls[0]["active_steps"], [24, 24])
        # The fresh bear differs from its original generator call only in IDs/seed.
        bm = probe.core.read_json(self.root / self.cfg["bear_source"] / "suite_manifest.json")
        old = probe.bear.expected_call(self.root / self.cfg["bear_source"], bm, 4)[1]
        new = probe.expected_call(self.output, self.m, "035_bear", 4)[1]
        self.assertEqual({k: v for k, v in old.items() if k != "subject"}, {k: v for k, v in new.items() if k != "case"})
        with self.assertRaisesRegex(RuntimeError, "Missing generation"):
            probe.records(self.output, self.m, ["035_bear"], complete=True)

    def test_car_controls_geometry_and_text_use_their_own_reference(self):
        source = self.root / self.cfg["road_source"]
        road = probe.core.read_json(source / "suite_manifest.json")
        for case in ("284_vehicle", "404_vehicle"):
            spec = self.m["cases"][case]
            original = road["source"]["source"]["cases"][case]
            self.assertEqual(spec["geometry"]["centroid_xy"], original["geometry"]["centroid_xy"])
            self.assertEqual(spec["geometry"]["K_analysis"], original["geometry"]["K_analysis"])
            self.assertEqual(spec["geometry"]["rendered_area"], original["rendered_object_area_pixels"])
            self.assertEqual(spec["appearance"]["subject_text"], "a red car")
            self.assertTrue(all("bear" not in t for t in spec["appearance"]["style_texts"].values()))
            self.assertEqual((self.output / f"inputs/{case}/edges.png").read_bytes(), (source / f"inputs/{case}/manual.png").read_bytes())

    def test_visually_annotated_car_fixtures_match_saved_real_masks(self):
        for case in ("284_vehicle", "404_vehicle"):
            folder = self.root / self.cfg["road_source"] / f"measurement/{case}/seed_2/manual_early"
            if not folder.exists():
                self.skipTest("Saved measurement masks unavailable")
            fixture = self.m["cases"][case]["fixture"]
            for label in ("subject", "road"):
                mask = np.asarray(Image.open(folder / f"{label}.png")) > 0
                self.assertTrue(score.original.known_points(mask, fixture[label])["passed"])

    def test_corrupt_control_and_source_overlap_are_refused(self):
        path = self.output / "inputs/035_bear/edges.png"
        before = path.read_bytes()
        try:
            path.write_bytes(b"damaged")
            with self.assertRaisesRegex(RuntimeError, "Changed completed"):
                probe.verified(self.output, self.cfg)
        finally:
            path.write_bytes(before)
        with self.assertRaisesRegex(RuntimeError, "separate scoring"):
            probe.prepare(self.cfg, self.root, self.root / self.cfg["bear_source"])

    def test_same_decision_on_saved_real_bear_scores(self):
        source = self.root / self.cfg["bear_source"]
        rows = [probe.core.read_json(source / f"measurement/seed_{s}/result.json") for s in range(4)]
        fixture = probe.core.read_json(source / "measurement/fixture/result.json")
        old = probe.tie.quality_tiebreak(probe.core.read_json(source / "selection.json"))
        result = score.decision(rows, fixture, self.m)
        self.assertEqual(result["selected_seed"], 1)
        self.assertEqual(result["candidates"], old["candidates"])
        failed = deepcopy(fixture)
        failed["selftest"]["road"]["passed"] = False
        self.assertIsNone(score.decision(rows, failed, self.m)["selected_seed"])

    def test_prepared_probe_contains_no_generated_or_selected_image(self):
        self.assertFalse((self.output / "generation").exists())
        self.assertFalse((self.output / "selection").exists())
        for case in self.cfg["cases"]:
            fixture = self.output / f"fixtures/{case}/image.png"
            self.assertEqual(probe.core.file_hash(fixture), self.m["cases"][case]["fixture"]["image_sha256"])


if __name__ == "__main__":
    unittest.main()
