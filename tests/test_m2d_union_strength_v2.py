"""CPU policy tests and replay of saved real inference; no external model mocks."""
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from scripts import m2d_subject_association_v2 as assoc
from scripts import probe_m2d_union_strength_v2 as probe


def box(identifier, coordinates, confidence):
    x0, y0, x1, y1 = coordinates
    return {"id": identifier, "query": "car", "box": coordinates,
            "score": confidence, "box_area": (x1 - x0) * (y1 - y0)}


class AgreementTests(unittest.TestCase):
    def test_competing_prominence_and_confidence_refuse(self):
        rows = [box(0, [0, 0, 10, 10], .8), box(1, [50, 50, 100, 100], .1)]
        result = assoc.evidence(rows, "car")
        self.assertEqual(result["frontier_ids"], [0, 1])
        self.assertFalse(result["agreement"])

    def test_dominated_background_detection_does_not_create_ambiguity(self):
        rows = [box(0, [0, 0, 10, 10], .1), box(1, [50, 50, 100, 100], .8)]
        result = assoc.evidence(rows, "car")
        self.assertEqual(result["frontier_ids"], [1])
        self.assertTrue(result["agreement"])

    def test_consistent_alternative_boxes_agree_without_iou_threshold(self):
        rows = [box(0, [0, 0, 20, 20], .4), box(1, [2, 2, 18, 18], .8)]
        self.assertTrue(assoc.evidence(rows, "car")["agreement"])
        self.assertFalse(assoc.evidence([], "car")["agreement"])

    def test_invalid_area_and_nonfinite_score_are_rejected(self):
        row = box(0, [0, 0, 20, 20], .4)
        for changed in (dict(row, box_area=1), dict(row, score=float("nan"))):
            with self.assertRaises(RuntimeError):
                assoc.evidence([changed], "car")

    def test_new_generation_calls_are_identical_to_v1(self):
        cfg, _, settings = probe.load_config(probe.DEFAULT_CONFIG)
        old_cfg, _, _ = probe.previous.load_config(probe.ROOT / cfg["measurement_source_config"])
        controls = {name: Image.new("RGB", (8, 8), i) for i, name in enumerate(("depth", "manual", "recovered"))}
        manifest = {"source": {"source": {"cases": {"404_vehicle": {"prompt": "a red car"}}, "model_revisions": {}}}}
        for row in probe.generation_rows(cfg):
            self.assertEqual(probe.expected_call(cfg, settings, manifest, controls, "404_vehicle", row["seed"], row["arm"]),
                             probe.previous.expected_call(old_cfg, settings, manifest, controls, "404_vehicle", row["seed"], row["arm"]))

    def test_real_replay_counts_and_unchanged_annotations(self):
        cfg, _, _ = probe.load_config(probe.DEFAULT_CONFIG)
        root = probe.ROOT / "measurements/m2d/recovered_union_strength_v1"
        if not (root / "measurement/archived").exists():
            self.skipTest("Returned real inference is not available")
        fixtures = probe.resume.read(probe.ROOT / cfg["association"]["fixtures"])["rows"]
        kept = refused = correct_points = refused_points = wrong_points = 0
        for path in sorted((root / "measurement/archived").glob("*/seed_*/*/result.json")):
            row = probe.resume.read(path)
            original = np.asarray(Image.open(path.parent / "subject.png")) > 0
            subject, ledger = assoc.adjudicate(original, row["association"], cfg["association"])
            if ledger["status"] == "selected_experimental":
                kept += 1
                self.assertIs(subject, original)  # retained actual mask, not a new inferred or fabricated mask
                self.assertEqual(ledger["selected_id"], row["association"]["selected_id"])
            else:
                refused += 1
                self.assertFalse(subject.any())
                self.assertIsNone(ledger["box"])
            fixture = next((f for f in fixtures if (f["case"], f["seed"], f["arm"]) == (row["case"], row["seed"], row["arm"])), None)
            if fixture:
                result = assoc.check_fixture(subject, ledger, fixture)
                self.assertTrue(result["passed"])
                correct_points += result["correct_instance_at_annotated_points"]
                refused_points += result["refused"]
                wrong_points += not result["correct_instance_at_annotated_points"] and not result["refused"]
        self.assertEqual((kept, refused), (14, 6))
        self.assertEqual((correct_points, refused_points, wrong_points), (8, 4, 0))


if __name__ == "__main__":
    unittest.main()
