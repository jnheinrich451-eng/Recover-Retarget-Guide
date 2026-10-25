"""Real returned controls and recorded model traces; no model mocks."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import probe_m2d_284_subject_edges as probe
from scripts import score_m2d_284_subject_edges as score


class SubjectEdgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = probe.ROOT / "measurements/m2d"
        cls.cfg = probe.configuration()
        cls.source = cls.root / cls.cfg["source_experiment"]
        if not (cls.source / "suite_manifest.json").exists():
            raise unittest.SkipTest("Returned 284 pair results unavailable")
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name) / "subject_edges"
        cls.m = probe.prepare(cls.cfg, cls.root, cls.output)
        cls.old = probe.base.verified(cls.source, probe.base.configuration())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_inputs_are_repeatable_and_only_subject_edges_are_added(self):
        self.assertEqual(self.m, probe.prepare(self.cfg, self.root, self.output))
        self.assertEqual(self.m, probe.verified(self.output, self.cfg))
        for key in ("settings", "model_revisions", "measurement_models", "scoring", "thresholds", "ground", "cases"):
            self.assertEqual(self.m[key], self.old[key])
        root = self.output / "inputs/284_vehicle"
        road = probe.pixels(root / "edges.png")
        subject = probe.pixels(root / "subject_edges.png")
        combined = probe.pixels(root / "combined_edges.png")
        mask = probe.pixels(root / "mask.png") > 0
        self.assertEqual(np.count_nonzero(combined != road), 2067)
        self.assertTrue(np.array_equal(combined[~mask], road[~mask]))
        self.assertTrue(np.array_equal(combined[mask], subject[mask]))
        self.assertFalse(subject[~mask].any())
        for name in ("depth", "mask", "reference", "edges"):
            self.assertEqual((root / f"{name}.png").read_bytes(),
                             (self.source / f"inputs/284_vehicle/{name}.png").read_bytes())
        self.assertFalse((self.output / "generation").exists())

    def test_generator_contract_matches_old_baseline_and_only_edges_change(self):
        for seed in self.cfg["seeds"]:
            _, plain = probe.expected_call(self.output, self.m, seed, "road_only")
            images, added = probe.expected_call(self.output, self.m, seed, "road_subject")
            self.assertEqual({k for k in plain if plain[k] != added[k]}, {"arm", "control_pixels_sha256"})
            self.assertEqual(plain["control_pixels_sha256"][0], added["control_pixels_sha256"][0])
            self.assertNotEqual(plain["control_pixels_sha256"][1], added["control_pixels_sha256"][1])
            self.assertEqual(added["control_pixels_sha256"], [probe.core.digest(np.asarray(im).tobytes()) for im in images])
            old = probe.base.expected_call(self.source, self.old, seed, "weaker_early")[1]
            self.assertEqual({k for k in old if old[k] != plain[k]}, {"arm"})
            self.assertEqual(added["control_vae_rng_seed"], seed)
        with self.assertRaisesRegex(RuntimeError, "Missing generation"):
            probe.records(self.output, self.m, complete=True)

    def test_pair_audit_on_real_changed_edge_records(self):
        root = self.root / "road_constraint_v1"
        rows = []
        for arm in ("recovered_early", "manual_early"):
            folder = root / f"generation/284_vehicle/seed_2/{arm}"
            call = probe.core.read_json(folder / "call.json")
            self.assertTrue(probe.timing.completed(folder, call))
            rows.append({"call": call, "result": probe.core.read_json(folder / "result.json"),
                         "runtime": probe.timing.native.verify_provenance(folder, root)[1]})
        self.assertTrue(all(probe.pair_checks(*rows).values()))
        self.assertFalse(probe.pair_checks(rows[0], rows[0])["different_edge_pixels"])
        bad = deepcopy(rows[1])
        bad["call"]["scales"][1] *= 2
        self.assertFalse(probe.pair_checks(rows[0], bad)["same_scales"])
        bad = deepcopy(rows[1])
        bad["result"]["actual_union_calls"]["2"]["encoded_control_sha256"] = "0" * 64
        self.assertFalse(probe.pair_checks(rows[0], bad)["same_encoded_mode_2"])

    def test_replay_refuses_changed_image_in_matching_runtime(self):
        row = self.m["baselines"]["4"]
        self.assertTrue(all(probe.replay_check(row, deepcopy(row)).values()))
        bad = deepcopy(row)
        bad["result"]["image_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "Road-only replay differs"):
            probe.replay_check(bad, row)
        # Changed runtime is recorded as a non-comparable historical replay;
        # the two fresh images must still pass strict within-pair checks.
        bad["runtime"]["gpu"] = "different runtime (metadata corruption test)"
        self.assertFalse(probe.replay_check(bad, row)["same_runtime"])

    def test_overlap_and_outside_mask_edges_refused(self):
        root = self.output / "inputs/284_vehicle"
        road = probe.pixels(root / "edges.png").copy()
        subject = probe.pixels(root / "subject_edges.png").copy()
        mask = probe.pixels(root / "mask.png") > 0
        bad_road = road.copy()
        bad_road[mask] = 255
        with self.assertRaisesRegex(RuntimeError, "Road crosses proxy"):
            probe.combine(bad_road, subject, mask)
        bad_subject = subject.copy()
        bad_subject[~mask] = 255
        with self.assertRaisesRegex(RuntimeError, "outside the aligned proxy"):
            probe.combine(road, bad_subject, mask)
        with self.assertRaisesRegex(RuntimeError, "separate scoring"):
            probe.prepare(self.cfg, self.root, self.source)

    def test_corrupt_control_refused(self):
        path = self.output / "inputs/284_vehicle/combined_edges.png"
        before = path.read_bytes()
        try:
            path.write_bytes(b"damaged")
            with self.assertRaisesRegex(RuntimeError, "Changed completed"):
                probe.verified(self.output, self.cfg)
        finally:
            path.write_bytes(before)

    def test_four_scoring_tasks_and_fixed_selector(self):
        tasks = list(score.tasks(self.output, self.m))
        self.assertEqual(len(tasks), 5)  # shared known-image fixture + four candidates
        self.assertEqual({(a, s) for a, s, _, _ in tasks if s is not None},
                         {(a, s) for a in self.cfg["arms"] for s in self.cfg["seeds"]})
        rows, fixture = score.previous.read_rows(self.source, self.old)
        selected = [r for r in rows if r["arm"] == "weaker_early" and r["seed"] in self.cfg["seeds"]]
        old = score.previous.decision(selected, fixture, self.old, "weaker_early")
        result = score.decision(selected, fixture, self.m, "road_only")
        self.assertEqual(result["candidates"], old["candidates"])
        self.assertIsNone(result["selected_seed"])
        self.assertEqual(result["absolute_quality_acceptance"], "not_measured")


if __name__ == "__main__":
    unittest.main()
