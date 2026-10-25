"""Pure boundary geometry and recorded-call checks; no external model mocks."""
from copy import deepcopy
import unittest

import cv2
import numpy as np
import yaml

from scripts import probe_m2d_road_constraint as probe
from scripts import score_m2d_road_constraint as score


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.zeros((64, 64), np.uint8)
        self.empty = np.zeros_like(self.labels, bool)
        self.names = {0: "grass", 1: "road", 2: "car"}
        self.cfg = {"route_labels": ["road"], "subject_labels": ["car"],
                    "neighbourhood": {"occluder_edge_margin_px": 1}}
        self.lines = [{"name": "left", "evidence": "visible", "points": [[16, 1], [16, 62]]}]

    def evaluate(self, labels=None, target=None, subject=None):
        return score.boundary_metrics(self.labels if labels is None else labels, self.names, self.cfg, self.lines,
                                      self.empty, self.empty if target is None else target,
                                      self.empty if subject is None else subject)

    def test_exact_trace_has_zero_distance(self):
        self.labels[:, 16:48] = 1
        result, _, _ = self.evaluate()
        self.assertEqual(result["trace_to_boundary"]["p95_px"], 0.)

    def test_straight_corridor_does_not_match_bent_trace(self):
        self.lines[0]["points"] = [[8, 1], [8, 20], [30, 40], [30, 62]]
        trace = score.audit.visible_traces(self.lines, self.labels.shape)
        for y in range(1, 63):
            x = np.flatnonzero(trace[y])[0]
            self.labels[y, x:58] = 1
        bent, _, _ = self.evaluate()
        straight = np.zeros_like(self.labels)
        straight[:, 30:58] = 1
        wrong, _, _ = self.evaluate(straight)
        self.assertEqual(bent["trace_to_boundary"]["median_px"], 0.)
        self.assertGreater(wrong["trace_to_boundary"]["p95_px"], bent["trace_to_boundary"]["p95_px"])

    def test_internal_island_boundary_is_retained(self):
        region = np.ones_like(self.empty)
        region[20:40, 20:40] = False
        boundary = score.semantic_boundary(region)
        self.assertTrue(boundary.any())
        self.assertTrue(boundary[19, 25])
        self.assertFalse(boundary[0].any() or boundary[-1].any() or boundary[:, 0].any() or boundary[:, -1].any())

    def test_subject_hole_edges_are_excluded(self):
        self.labels[:, 16:48] = 1
        self.labels[20:40, 16:32] = 2
        _, _, boundary = self.evaluate()
        self.assertFalse(boundary[19:41, 15:33].any())
        self.assertTrue(boundary[5, 16])

    def test_domain_cut_does_not_create_new_boundary(self):
        self.labels[:] = 1
        target = self.empty.copy()
        target[10:50, 10:50] = True
        _, _, boundary = self.evaluate(target=target)
        self.assertFalse(boundary.any())

    def test_generated_subject_does_not_shrink_fixed_query(self):
        self.labels[:, 16:48] = 1
        baseline, trace, _ = self.evaluate()
        subject = self.empty.copy()
        subject[10:50, 12:24] = True
        changed, hidden_trace, _ = self.evaluate(subject=subject)
        self.assertTrue(np.array_equal(trace, hidden_trace))
        self.assertEqual(baseline["fixed_query_sha256"], changed["fixed_query_sha256"])
        self.assertGreater(changed["overlapped_by_generated_subject_pixels"], 0)

    def test_empty_road_is_unmeasured_not_zero_error(self):
        result, _, _ = self.evaluate()
        self.assertEqual(result["trace_to_boundary"]["status"], "empty_target")
        self.assertIsNone(result["trace_to_boundary"]["median_px"])

    def test_fully_occluded_query_is_unmeasured(self):
        self.labels[:, 16:48] = 1
        result, _, _ = self.evaluate(target=np.ones_like(self.empty))
        self.assertEqual(result["trace_to_boundary"]["status"], "empty_query")

    def test_nonvisible_strokes_are_not_scored(self):
        self.labels[:, 16:48] = 1
        self.lines.append({"name": "inferred", "evidence": "occluded", "points": [[1, 1], [1, 62]]})
        result, _, _ = self.evaluate()
        self.assertEqual(len(result["per_visible_stroke"]), 1)

    def test_misaligned_mask_and_unknown_labels_refused(self):
        with self.assertRaisesRegex(RuntimeError, "coordinates"):
            self.evaluate(target=np.zeros((32, 32), bool))
        self.labels[:] = 250
        with self.assertRaisesRegex(RuntimeError, "Unknown semantic"):
            self.evaluate()


class RecordedExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.parent, cls.settings = probe.load_config(probe.DEFAULT_CONFIG)
        cls.root = probe.ROOT / "measurements/m2d/recovered_union_strength_v2"
        if not (cls.root / "suite_manifest.json").exists():
            raise unittest.SkipTest("Returned real outputs unavailable")
        cls.old = probe.resume.read(cls.root / "suite_manifest.json")
        cls.manifest = {"source": cls.old["source"], "scales": cls.parent["arms"]["lower_edges"],
                        "depth_active_steps": cls.parent["active_steps"]}

    def test_baseline_replays_the_four_actual_recorded_calls(self):
        for case in self.cfg["cases"]:
            controls = probe.source.controls_for(self.root, case)
            for seed in self.cfg["seeds"]:
                call = probe.expected_call(self.cfg, self.settings, self.manifest, controls, case, seed, "recovered_early")
                old = probe.resume.read(self.root / f"generation/{case}/seed_{seed}/lower_edges/call.json")
                for key, value in call.items():
                    if key != "arm":
                        self.assertEqual(value, old[key], (case, seed, key))

    def test_two_factors_change_only_edge_source_or_duration(self):
        for case in self.cfg["cases"]:
            controls = probe.source.controls_for(self.root, case)
            calls = {arm: probe.expected_call(self.cfg, self.settings, self.manifest, controls, case, 0, arm) for arm in self.cfg["arms"]}
            baseline = calls["recovered_early"]
            for arm, call in calls.items():
                for key in set(call) - {"arm", "active_steps", "control_guidance_end", "control_pixels_sha256"}:
                    self.assertEqual(call[key], baseline[key], (arm, key))
                self.assertEqual(call["active_steps"][0], 12)
                self.assertEqual(call["control_pixels_sha256"][0], baseline["control_pixels_sha256"][0])
                same_edge = call["control_pixels_sha256"][1] == baseline["control_pixels_sha256"][1]
                self.assertEqual(same_edge, arm.startswith("recovered"))
                self.assertEqual(call["active_steps"][1], 24 if arm.endswith("full") else 12)
        self.assertEqual(len(probe.generation_rows(self.cfg)), 16)

    def test_pair_audit_on_real_calls_detects_corrupted_noise(self):
        path = self.root / "generation/284_vehicle/seed_0/lower_edges"
        record = probe.generation_record(path, self.root)
        self.assertTrue(all(probe.comparison_checks(record, record, True).values()))
        self.assertFalse(all(probe.comparison_checks(record, record, False).values()))
        changed = deepcopy(record)
        changed[0]["initial_latents_sha256"] = "corrupt-test-ledger"
        self.assertFalse(probe.comparison_checks(record, changed, True)["same_initial_latents"])

    def test_runtime_mismatch_is_reported_before_inference(self):
        path = self.root / "generation/284_vehicle/seed_0/lower_edges"
        _, runtime = probe.native.verify_provenance(path, self.root)
        probe.require_runtime(runtime, runtime, "Generator")
        with self.assertRaisesRegex(RuntimeError, "new output directory alone cannot fix"):
            probe.require_runtime(dict(runtime, gpu="changed-test-ledger"), runtime, "Generator")

    def test_real_trace_domain_is_fixed_across_all_28_saved_outputs(self):
        suite = yaml.safe_load(probe.source.general.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        lines = {c["id"]: c.get("polylines", []) for c in suite["cases"]}
        original = self.old["source"]["source"]
        names = {int(k): v.strip() for k, v in original["labels"]["id2label"].items()}
        for case in self.cfg["cases"]:
            hashes = set()
            inputs = self.root / "inputs" / case
            for path in (self.root / "measurement/strength" / case).glob("seed_*/*/labels.png"):
                labels = np.asarray(score.Image.open(path))
                result, _, _ = score.boundary_metrics(labels, names, original["ground_config"], lines[case],
                    score.prior.mask(inputs / "source_occluders.png"), score.prior.mask(inputs / "target_visible.png"),
                    score.prior.mask(path.parent / "subject.png"))
                hashes.add(result["fixed_query_sha256"])
                self.assertGreater(result["fixed_query_pixels"], 0)
            self.assertEqual(len(hashes), 1)


if __name__ == "__main__":
    unittest.main()
