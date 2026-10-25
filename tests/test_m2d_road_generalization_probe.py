"""CPU tests of census/routing/statistics, not mocks of the segmenter or FLUX."""
import unittest

import numpy as np
import yaml

from scripts import probe_m2d_road_generalization as probe


class GeneralizationProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = yaml.safe_load(probe.DEFAULT_CONFIG.read_text())

    def test_case_records_and_generation_budget(self):
        total = 0
        unique = set()
        for case in self.suite["cases"]:
            placement, qc, paths = probe.records(case)
            self.assertTrue(all(p.is_file() for p in paths))
            self.assertGreaterEqual(placement["t_a"], 0)
            self.assertEqual(np.asarray(qc["K_1024"]).shape, (3, 3))
            unique.add((case["clip"], placement["t_a"]))
            if case["generation"]:
                baseline = probe.read(probe.ROOT / f"measurements/m2d_anchors/generation_{case['clip'][9:]}_{case['category']}.json")
                self.assertEqual(baseline["clip"], case["clip"])
                self.assertEqual(baseline["category"], case["category"])
                self.assertEqual(baseline["polarity"], "near_bright")
                self.assertEqual(len(baseline["upstream_hashes"]["c3"]), 64)
                total += (2 if case.get("polylines") else 1) * len(self.suite["default_seeds"])
        self.assertEqual(total, 16)
        self.assertEqual(len(unique), 7)  # the human/vehicle pair shares one frame

    def test_no_road_and_ambiguous_cases_have_no_road_strokes(self):
        for case in self.suite["cases"]:
            trace = probe.trace_for(case)
            raster = probe.core.draw_road(trace, 2)
            if case["reference_class"] not in ("paved_road", "dirt_track"):
                self.assertFalse(raster.any())
            for line in trace["polylines"]:
                self.assertNotIn("marking", line["name"])
                xy = np.asarray(line["points"])
                self.assertTrue(((xy >= 0) & (xy < 1024)).all())
        bad = {"reference_class": "no_clear_road", "polylines": [{"name": "invented"}]}
        with self.assertRaises(RuntimeError):
            probe.trace_for(bad)

    def test_negative_scene_prompts_do_not_request_a_road(self):
        for case in self.suite["cases"]:
            if case["generation"] and case["reference_class"] in ("no_clear_road", "ambiguous_trail"):
                for word in ("road", "highway", "asphalt", "lane", "track"):
                    self.assertNotIn(word, case["prompt"].lower())

    def test_statistics_on_known_label_map(self):
        labels = np.zeros((8, 8), dtype=np.uint8)
        labels[1:3, 1:3] = 6
        labels[6, 6] = 6
        probabilities = np.full((8, 8), 0.25, dtype=np.float32)
        stats = probe.label_stats(labels, probabilities, 6)
        self.assertEqual(stats["pixels"], 5)
        self.assertEqual(stats["largest_component_pixels"], 4)
        self.assertEqual(stats["area_fraction"], 5/64)
        self.assertEqual(stats["mean_probability"], 0.25)
        empty = probe.label_stats(labels, probabilities, 52)
        self.assertEqual(empty["pixels"], 0)
        self.assertEqual(empty["largest_component_pixels"], 0)
        self.assertIn("unreviewed", stats["presence_verdict"])


if __name__ == "__main__":
    unittest.main()
