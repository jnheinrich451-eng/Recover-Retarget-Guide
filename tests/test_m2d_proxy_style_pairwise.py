"""Decision/provenance contracts using explicit labels, not mocked VLM inference.

Real archived images exercise preparation. These tests do not establish Qwen's
ability to rank photographic appearance; the Colab probe measures that.
"""
from argparse import Namespace
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts import probe_m2d_proxy_style_pairwise as probe


def observation(label):
    return None if label is None else {"preference": label, "evidence": "Decision test data, not model output"}


def views(ab="A", ba="B", crop_ab=None, crop_ba=None):
    return {"full": {"ab": observation(ab), "ba": observation(ba)},
            "subject_crop": {"ab": observation(ab if crop_ab is None else crop_ab),
                             "ba": observation(ba if crop_ba is None else crop_ba)}}


def pair(a, b, outcome):
    return {"seeds": [a, b], "outcome": outcome}


class ComparisonContracts(unittest.TestCase):
    def test_swapping_position_keeps_candidate_identity(self):
        self.assertEqual(probe.order_consensus("A", "B")["outcome"], "a")
        self.assertEqual(probe.order_consensus("B", "A")["outcome"], "b")

    def test_always_choose_first_or_second_is_unresolved(self):
        for label in ("A", "B"):
            self.assertEqual(probe.order_consensus(label, label)["reason"], "presentation_order_disagreement")

    def test_only_explicit_consistent_ties_get_half_credit(self):
        self.assertEqual(probe.order_consensus("tie", "tie")["outcome"], "tie")
        for first, second in (("tie", "A"), ("uncertain", "uncertain"), (None, None), (None, "B")):
            self.assertEqual(probe.order_consensus(first, second)["outcome"], "unresolved")

    def test_crop_disagreement_cannot_be_averaged_into_a_tie(self):
        result = probe.pair_consensus(views("A", "B", "B", "A"))
        self.assertEqual(result["reason"], "full_crop_disagreement")
        self.assertEqual(result["outcome"], "unresolved")

    def test_missing_or_uncertain_view_is_unresolved(self):
        self.assertEqual(probe.pair_consensus({"full": views()["full"]})["outcome"], "unresolved")
        self.assertEqual(probe.pair_consensus(views(crop_ab="uncertain"))["outcome"], "unresolved")

    def test_pair_views_count_as_one_result(self):
        self.assertEqual(probe.pair_consensus(views())["outcome"], "a")

    def test_all_loss_still_zero_not_missing(self):
        result = probe.grade([4, 5, 6], [pair(4, 5, "a"), pair(4, 6, "a"), pair(5, 6, "a")], True)
        self.assertEqual([r["complete_score"] for r in result["candidates"]], [100, 50, 0])
        self.assertEqual(result["ranked_seed_groups"], [[4], [5], [6]])
        self.assertIsNone(result["selected_seed"])
        self.assertFalse(result["automatic_reconstruction_authorized"])

    def test_all_ties_preserve_equal_rank_without_seed_tiebreak(self):
        result = probe.grade([5, 4, 6], [pair(4, 5, "tie"), pair(4, 6, "tie"), pair(5, 6, "tie")], True)
        self.assertEqual(result["ranked_seed_groups"], [[4, 5, 6]])
        self.assertEqual([r["complete_score"] for r in result["candidates"]], [50, 50, 50])

    def test_sparse_100_percent_does_not_beat_complete_lower_score(self):
        result = probe.grade([4, 5, 6], [pair(4, 5, "a"), pair(4, 6, "unresolved"), pair(5, 6, "a")], True)
        self.assertEqual(result["status"], "unresolved_comparisons")
        self.assertEqual(result["top_relative_seeds"], [])
        row = result["candidates"][0]
        self.assertEqual(row["resolved_preference_score"], 100)
        self.assertIsNone(row["complete_score"])
        self.assertEqual(row["score_bounds"], [50, 100])

    def test_all_unresolved_is_not_zero_realism(self):
        result = probe.grade([4, 5], [pair(4, 5, "unresolved")], True)
        self.assertIsNone(result["candidates"][0]["resolved_preference_score"])
        self.assertEqual(result["candidates"][0]["score_bounds"], [0, 100])

    def test_control_failure_withholds_ranking_but_keeps_measurements(self):
        result = probe.grade([4, 5], [pair(4, 5, "a")], False)
        self.assertEqual(result["status"], "instrument_check_failed")
        self.assertEqual(result["ranked_seed_groups"], [])
        self.assertEqual(result["candidates"][0]["complete_score"], 100)

    def test_three_way_cycle_reported_without_forced_winner(self):
        result = probe.grade([4, 5, 6], [pair(4, 5, "a"), pair(4, 6, "b"), pair(5, 6, "a")], True)
        self.assertEqual(result["status"], "cyclic_preferences")
        self.assertTrue(result["preference_cycle_detected"])
        self.assertEqual(result["top_relative_seeds"], [])

    def test_four_way_cycle_is_detected(self):
        pairs = [pair(4, 5, "a"), pair(4, 6, "tie"), pair(4, 7, "b"),
                 pair(5, 6, "a"), pair(5, 7, "tie"), pair(6, 7, "a")]
        self.assertTrue(probe.grade([4, 5, 6, 7], pairs, True)["preference_cycle_detected"])

    def test_missing_duplicate_foreign_pairs_refused(self):
        for pairs in ([], [pair(4, 5, "a")] * 2, [pair(4, 6, "a")]):
            with self.assertRaisesRegex(RuntimeError, "candidate pair"):
                probe.grade([4, 5], pairs, True)


class AnswerContracts(unittest.TestCase):
    def setUp(self):
        self.cfg = probe.config(probe.DEFAULT)
        self.inputs = [{"item": "seed_4", "sha256": "a", "pixels_wh": [1024, 1024]},
                       {"item": "seed_5", "sha256": "b", "pixels_wh": [1024, 1024]}]

    def attempt(self, raw, retry=False):
        obj, err = probe.parse(raw)
        return {"raw": raw, "observation": obj, "parse_error": err, "inputs": self.inputs,
                "prompt_sha256": probe.core.digest(probe.question(self.cfg, retry).encode())}

    def test_invalid_json_never_becomes_tie_or_rejection(self):
        for raw in ('{"preference":"A"}', '{"preference":1,"evidence":"x"}',
                    '{"preference":"A","evidence":""}', '[]', 'A',
                    '{"preference":"A","evidence":"x","score":99}'):
            self.assertEqual(probe.parse(raw), (None, "invalid_structured_answer"))

    def test_valid_uncertain_is_terminal_not_retried(self):
        a = self.attempt(json.dumps(observation("uncertain")))
        self.assertEqual(probe.finish([a], self.cfg, self.inputs)["observation"]["preference"], "uncertain")
        with self.assertRaisesRegex(RuntimeError, "Cannot retry"):
            probe.finish([a, self.attempt(json.dumps(observation("A")), True)], self.cfg, self.inputs)

    def test_one_format_retry_then_missing_observation(self):
        a = self.attempt("bad")
        with self.assertRaisesRegex(RuntimeError, "unfinished"):
            probe.finish([a], self.cfg, self.inputs)
        result = probe.finish([a, self.attempt("still bad", True)], self.cfg, self.inputs)
        self.assertIsNone(result["observation"])

    def test_swapped_or_changed_images_are_refused(self):
        a = self.attempt(json.dumps(observation("A")))
        with self.assertRaisesRegex(RuntimeError, "Image order"):
            probe.finish([a], self.cfg, self.inputs[::-1])
        broken = copy.deepcopy(a)
        broken["observation"]["preference"] = "B"
        with self.assertRaisesRegex(RuntimeError, "raw output"):
            probe.finish([broken], self.cfg, self.inputs)

    def test_model_message_has_two_images_and_no_seed_grade_or_expected_label(self):
        msg = probe.messages(probe.question(self.cfg))[0]["content"]
        self.assertEqual([m["type"] for m in msg], ["text", "image", "text", "image", "text"])
        text = json.dumps(msg)
        for forbidden in ("seed_5", "nps_brown", "expected", "cartoon_control", "6 may outrank"):
            self.assertNotIn(forbidden, text)

    def test_plan_exhaustive_and_orders_are_identity_bound(self):
        plan = probe.build_plan([4, 5, 6, 7], self.cfg)
        self.assertEqual(len(plan), 9)
        self.assertEqual(len([p for p in plan if p["kind"] == "candidate"]), 6)
        manifest = {"inputs": {s: {"views": {"full": {"sha256": s}}} for s in ("seed_4", "seed_5")}}
        p = next(p for p in plan if p["kind"] == "candidate")
        self.assertEqual(probe.input_record(manifest, p, "full", "ab"),
                         list(reversed(probe.input_record(manifest, p, "full", "ba"))))


class ActualArchivedInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = probe.ROOT / "measurements/m2d"
        cls.photos = cls.root / "035_bear_style_selftest_inputs_v1"
        cls.source = cls.root / "035_bear_style_filter_v1"
        if not (cls.source / "index.json").exists() or not (cls.photos / "usfs_brown.jpg").exists():
            raise unittest.SkipTest("Returned style run / exact photographic fixtures absent")

    def test_prepare_exact_images_resume_and_detect_modified_input_without_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            known = root / "known"
            args = Namespace(config=probe.fixtures.DEFAULT, probe_root=self.root, output_dir=known,
                             photo_dir=self.photos, device="cpu", gpu_memory_gib=None)
            probe.fixtures.prepare(args)
            cfg = probe.config(probe.DEFAULT)
            cfg["fixture_run"] = str(known)
            args.output_dir = root / "pairwise"
            before = probe.core.file_hash(self.source / "index.json")
            manifest, baseline = probe.prepare(args, cfg)
            again, _ = probe.prepare(args, cfg)
            self.assertEqual(manifest, again)
            self.assertEqual(len(manifest["plan"]), 9)
            self.assertEqual(baseline["status"], "no_style_supported_candidate")
            self.assertEqual(before, probe.core.file_hash(self.source / "index.json"))
            self.assertFalse((args.output_dir / "results.json").exists())
            self.assertFalse((args.output_dir / "model.json").exists())
            for seed in (4, 5, 6, 7):
                for view in cfg["views"]:
                    self.assertEqual(probe.core.file_hash(args.output_dir / f"inputs/seed_{seed}/{view}.png"),
                                     probe.core.file_hash(self.source / f"seed_{seed}/{view}.png"))
            # A changed question cannot reuse a partial experiment.
            changed = copy.deepcopy(cfg)
            changed["subject"] = "panther"
            with self.assertRaisesRegex(RuntimeError, "Existing artifact differs"):
                probe.prepare(args, changed)
            # A changed copied image cannot silently resume.
            (args.output_dir / "inputs/seed_4/full.png").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "Existing artifact differs"):
                probe.prepare(args, cfg)

    def test_cached_photo_crop_must_still_belong_to_the_pinned_photo(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = Namespace(config=probe.fixtures.DEFAULT, probe_root=self.root, output_dir=root / "known",
                             photo_dir=self.photos, device="cpu", gpu_memory_gib=None)
            probe.fixtures.prepare(args)
            cfg = probe.config(probe.DEFAULT)
            cfg["fixture_run"] = str(args.output_dir)
            crop = args.output_dir / "nps_brown/subject_crop.png"
            # Use another REAL image, not synthetic photographic model evidence.
            crop.write_bytes((self.source / "seed_5/subject_crop.png").read_bytes())
            args.output_dir = root / "pairwise"
            with self.assertRaisesRegex(RuntimeError, "known source"):
                probe.prepare(args, cfg)

    def test_missing_entire_selftest_uses_repository_zip_without_creating_old_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = probe.config(probe.DEFAULT)
            known = root / "no_selftest"
            cfg["fixture_run"] = str(known)
            args = Namespace(probe_root=self.root, output_dir=root / "pairwise", device="cpu", gpu_memory_gib=None)
            manifest, baseline = probe.prepare(args, cfg)
            self.assertEqual(manifest, probe.prepare(args, cfg)[0])
            self.assertFalse(known.exists())
            self.assertFalse(manifest["fixture_results_used"])
            self.assertIsNone(manifest["fixture_manifest_sha256"])
            self.assertIsNone(manifest["fixture_index_sha256"])
            self.assertFalse((args.output_dir / "results.json").exists())
            self.assertEqual(baseline["status"], "no_style_supported_candidate")
            for item in ("nps_brown", "usfs_brown"):
                self.assertEqual(manifest["inputs"][item]["photo_origin"]["kind"], "offline_zip")
                self.assertEqual(probe.core.file_hash(args.output_dir / f"inputs/{item}/source.jpg"),
                                 probe.core.file_hash(self.photos / f"{item}.jpg"))
                self.assertTrue((args.output_dir / f"inputs/{item}/subject_crop.png").is_file())

    def test_partial_selftest_missing_source_and_crop_recovers_without_mutating_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            known = root / "known"
            args = Namespace(config=probe.fixtures.DEFAULT, probe_root=self.root, output_dir=known,
                             photo_dir=self.photos, device="cpu", gpu_memory_gib=None)
            probe.fixtures.prepare(args)
            removed = [known / "usfs_brown/source.jpg", known / "usfs_brown/subject_crop.png"]
            original_hashes = {p.name: probe.core.file_hash(p) for p in removed}
            for p in removed:
                p.unlink()
            prior = {p.relative_to(known).as_posix(): probe.core.file_hash(p) for p in known.rglob("*") if p.is_file()}
            cfg = probe.config(probe.DEFAULT)
            cfg["fixture_run"] = str(known)
            args.output_dir = root / "pairwise"
            manifest, _ = probe.prepare(args, cfg)
            self.assertEqual(manifest, probe.prepare(args, cfg)[0])
            self.assertEqual(manifest["inputs"]["nps_brown"]["photo_origin"]["kind"], "selftest_cache")
            self.assertEqual(manifest["inputs"]["usfs_brown"]["photo_origin"]["kind"], "offline_zip")
            for p in removed:
                self.assertFalse(p.exists())
                self.assertEqual(probe.core.file_hash(args.output_dir / "inputs/usfs_brown" / p.name),
                                 original_hashes[p.name])
            after = {p.relative_to(known).as_posix(): probe.core.file_hash(p) for p in known.rglob("*") if p.is_file()}
            self.assertEqual(prior, after)

    def test_present_bad_photo_is_not_silently_replaced_by_valid_zip(self):
        with tempfile.TemporaryDirectory() as temp:
            known = Path(temp) / "known"
            fixture = probe.yaml.safe_load(probe.fixtures.DEFAULT.read_text(encoding="utf-8"))["fixtures"][0]
            original = known / fixture["id"] / "source.jpg"
            original.parent.mkdir(parents=True)
            original.write_bytes((self.photos / "usfs_brown.jpg").read_bytes())
            archive = self.photos / "style_fixture_photos.zip"
            with self.assertRaisesRegex(RuntimeError, "Photo bytes changed"):
                probe.photo_inputs(known, fixture, archive)

    def test_missing_or_invalid_zip_has_actionable_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = probe.yaml.safe_load(probe.fixtures.DEFAULT.read_text(encoding="utf-8"))["fixtures"][0]
            archive = root / "photos.zip"
            with self.assertRaisesRegex(RuntimeError, "--photo-zip"):
                probe.photo_inputs(root / "absent", fixture, archive)
            with probe.zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(fixture["local_name"], (self.photos / "usfs_brown.jpg").read_bytes())
            with self.assertRaisesRegex(RuntimeError, "Photo bytes changed"):
                probe.photo_inputs(root / "absent", fixture, archive)


if __name__ == "__main__":
    unittest.main()
