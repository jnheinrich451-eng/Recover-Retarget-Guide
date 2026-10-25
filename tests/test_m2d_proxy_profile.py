"""Profile contracts and selection policy; no external-model substitutes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts import probe_m2d_proxy_profile as profile
from scripts import probe_m2d_proxy_scores as score


def observation(facing="right", view="side", **changes):
    return {"facing": facing, "view": view, "body_complete": True,
            "head_aligned": True, "evidence": "synthetic policy input", **changes}


class RuntimeDeviceMapTests(unittest.TestCase):
    """Real Torch tensors/Accelerate dispatch; no substitute Qwen model."""

    def test_single_device_without_huggingface_attribute(self):
        import torch
        model = torch.nn.Linear(2, 2)
        self.assertFalse(hasattr(model, "hf_device_map"))
        self.assertEqual(profile.runtime_device_map(model), {"": "cpu"})

    def test_preserves_accelerate_dispatch_map(self):
        import torch
        from accelerate import dispatch_model
        model = dispatch_model(torch.nn.Linear(2, 2), device_map={"": "cpu"})
        self.assertEqual(profile.runtime_device_map(model),
                         {k: str(v) for k, v in model.hf_device_map.items()})

    def test_buffer_devices_are_not_hidden_by_first_parameter(self):
        import torch
        model = torch.nn.Linear(2, 2)
        model.register_buffer("placement_fixture", torch.empty(1, device="meta"))
        self.assertEqual(profile.runtime_device_map(model),
                         {"weight": "cpu", "bias": "cpu", "placement_fixture": "meta"})


class ProfileTests(unittest.TestCase):
    def test_both_registered_profile_configs_load(self):
        for name in ("m2d_proxy_profile.yaml", "m2d_proxy_profile_landmarks.yaml", "m2d_proxy_profile_scalar.yaml"):
            cfg = profile.config(profile.ROOT / "configs" / name)
            self.assertEqual(cfg["canonical_facing"], "right")
            self.assertIsInstance(cfg["prompt"], str)

    def test_parse_only_explicit_structured_answers(self):
        obj = observation()
        self.assertEqual(profile.parse_answer(json.dumps(obj)), (obj, None))
        self.assertEqual(profile.parse_answer("```json\n" + json.dumps(obj) + "\n```"), (obj, None))
        for value in ("right", "{}", "[]", "null", json.dumps({**obj, "body_complete": "true"}),
                      json.dumps({**obj, "facing": "right-ish"}), json.dumps({**obj, "extra": 1})):
            with self.subTest(value=value):
                self.assertIsNone(profile.parse_answer(value)[0])

    def test_only_consistent_right_side_view_is_supported(self):
        result = profile.assess(observation(), observation("left"))
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["observed_facing"], "right")
        left = profile.assess(observation("left"), observation("right"))
        self.assertEqual(left["status"], "noncompliant")
        self.assertIn("not_right_facing", left["reason"])

    def test_landmark_direction_uses_coordinates_not_a_text_facing_claim(self):
        obj = observation()
        del obj["facing"]
        obj.update(snout_xy=[800, 300], rump_xy=[200, 500])
        a, error = profile.parse_answer(json.dumps(obj), "m2d_proxy_profile_v2")
        self.assertIsNone(error)
        self.assertEqual(a["facing"], "right")
        self.assertAlmostEqual(a["snout_rump_dx"], .6)
        obj.update(snout_xy=[200, 300], rump_xy=[800, 500])
        b, error = profile.parse_answer(json.dumps(obj), "m2d_proxy_profile_v2")
        self.assertEqual(profile.assess(a, b)["status"], "supported")
        for bad in ([1100, 500], [True, 200], [float("nan"), 200], [100]):
            obj["snout_xy"] = bad
            self.assertIsNone(profile.parse_answer(json.dumps(obj), "m2d_proxy_profile_v2")[0])
        obj["snout_xy"] = None
        self.assertEqual(profile.parse_answer(json.dumps(obj), "m2d_proxy_profile_v2")[0]["facing"], "uncertain")

    def test_oblique_and_cropped_are_known_noncompliance(self):
        for changes in ({"view": "oblique"}, {"body_complete": False}, {"head_aligned": False}):
            result = profile.assess(observation(**changes), observation("left", **changes))
            self.assertEqual(result["status"], "noncompliant")

    def test_uncertain_and_mirror_disagreement_never_pass(self):
        for a, b in ((None, observation("left")), (observation(), observation()),
                     (observation(body_complete=None), observation("left")),
                     (observation(), observation("left", "oblique")),
                     (observation("uncertain"), observation("left"))):
            self.assertEqual(profile.assess(a, b)["status"], "uncertain")


class ScalarProfileTests(unittest.TestCase):
    """Artificial JSON inputs test contracts, never substitute model predictions."""

    def setUp(self):
        self.cfg = profile.config(profile.ROOT / "configs/m2d_proxy_profile_scalar.yaml")
        self.answer = {"snout_x": 800, "rump_x": 200, "view": "side", "body_complete": True,
                       "head_aligned": True, "evidence": "artificial parser input"}

    def attempt(self, obj):
        raw = json.dumps(obj)
        parsed, error = profile.parse_answer(raw, self.cfg["schema"])
        return {"raw": raw, "observation": parsed, "parse_error": error}

    def test_scalar_schema_accepts_only_finite_numbers_or_null(self):
        self.assertEqual(self.attempt(self.answer)["observation"]["facing"], "right")
        for bad in ([800, 200], [1, 2, 3, 4], "800", True, {}, -1, 1001, float("nan"), float("inf")):
            with self.subTest(bad=bad):
                self.assertEqual(self.attempt({**self.answer, "snout_x": bad})["parse_error"], "invalid_structured_answer")
        unknown = self.attempt({**self.answer, "snout_x": None})
        self.assertIsNone(unknown["parse_error"])
        self.assertEqual(unknown["observation"]["facing"], "uncertain")
        equal = self.attempt({**self.answer, "snout_x": 200})
        self.assertEqual(equal["observation"]["facing"], "uncertain")

    def test_consistency_does_not_mean_requested_direction(self):
        for original, mirror, desired, verdict in (("left", "right", "mismatch", "noncompliant"),
                                                   ("right", "left", "matches", "supported")):
            result = profile.assess(observation(original), observation(mirror), self.cfg["schema"])
            self.assertTrue(result["mirror_consistent"])
            self.assertEqual(result["mirror_status"], "consistent")
            self.assertEqual(result["requested_direction_status"], desired)
            self.assertEqual(result["status"], verdict)

    def test_invalid_mirror_is_unknown_without_erasing_original_direction(self):
        result = profile.assess(observation(), None, self.cfg["schema"])
        self.assertIsNone(result["mirror_consistent"])
        self.assertEqual(result["mirror_status"], "unknown")
        self.assertEqual(result["original_facing"], "right")
        self.assertEqual(result["requested_direction_status"], "matches")
        self.assertEqual(result["status"], "uncertain")

    def test_oblique_and_mirror_conflict_are_distinct(self):
        oblique = profile.assess(observation(view="oblique"), observation("left", "oblique"), self.cfg["schema"])
        self.assertEqual(oblique["mirror_status"], "consistent")
        self.assertEqual(oblique["profile_suitability"], "noncompliant")
        self.assertEqual(oblique["status"], "noncompliant")
        conflict = profile.assess(observation(), observation(), self.cfg["schema"])
        self.assertEqual(conflict["mirror_status"], "inconsistent")
        self.assertEqual(conflict["status"], "uncertain")

    def test_valid_left_or_uncertain_never_retries(self):
        for change in ({"snout_x": 100}, {"snout_x": None}, {"view": "uncertain"}, {"view": "oblique"}):
            attempts = [self.attempt({**self.answer, **change})]
            self.assertFalse(profile.format_retry_needed(attempts, self.cfg))
            self.assertEqual(profile.finish_attempts(attempts, self.cfg)["format_retry_count"], 0)
            with self.assertRaisesRegex(RuntimeError, "Cannot retry a valid answer"):
                profile.finish_attempts(attempts + [self.attempt(self.answer)], self.cfg)

    def test_one_format_retry_is_bounded_and_both_answers_are_retained(self):
        invalid = self.attempt({**self.answer, "snout_x": [1, 2, 3, 4]})
        self.assertTrue(profile.format_retry_needed([invalid], self.cfg))
        with self.assertRaisesRegex(RuntimeError, "unfinished"):
            profile.finish_attempts([invalid], self.cfg)
        for final in (invalid, self.attempt(self.answer)):
            attempts = [invalid, final]
            self.assertFalse(profile.format_retry_needed(attempts, self.cfg))
            result = profile.finish_attempts(attempts, self.cfg)
            self.assertEqual(result["attempts"], attempts)
            self.assertEqual(result["format_retry_count"], 1)
        with self.assertRaisesRegex(RuntimeError, "count"):
            profile.finish_attempts([invalid] * 3, self.cfg)

    def test_saved_attempts_can_resume_without_changing_the_raw_answer(self):
        attempts = [self.attempt({}), self.attempt(self.answer)]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "attempt.json"
            profile.save(path, attempts[0])
            resumed = [profile.core.read_json(path), attempts[1]]
            self.assertEqual(profile.finish_attempts(resumed, self.cfg), profile.finish_attempts(attempts, self.cfg))
        attempts[-1]["observation"]["facing"] = "left"
        with self.assertRaisesRegex(RuntimeError, "differs from raw"):
            profile.finish_attempts(attempts, self.cfg)

    def test_new_score_config_changes_only_profile_measurement(self):
        old = score.load_config(score.ROOT / "configs/m2d_proxy_scores_right_profile.yaml")
        new = score.load_config(score.ROOT / "configs/m2d_proxy_scores_right_profile_v3.yaml")
        for cfg in (old, new):
            for key in ("predictions", "profile_config", "profile_spec"):
                cfg.pop(key)
        self.assertEqual(old, new)

    def test_report_uses_unknown_for_missing_mirror_evidence(self):
        row = {"seed": 0, "status": "uncertain", "observations": {"original": observation(), "mirrored": None}}
        columns = profile.summary_columns(row)
        self.assertEqual(columns["original_facing"], "right")
        self.assertEqual(columns["mirror_status"], "unknown")
        self.assertIn("Original matches requested right", profile.summary_table([row]))
        with tempfile.TemporaryDirectory() as td:
            score.raw_review(Path(td), [{"seed": 0, "profile": row}])
            rendered = (Path(td) / "review.html").read_text()
            self.assertIn("Mirror consistency", rendered)
            self.assertIn("<td>unknown</td>", rendered)


class RightProfileSelectionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = score.load_config(score.ROOT / "configs/m2d_proxy_scores_right_profile.yaml")
        self.policy = score.policy_template(self.cfg)
        self.policy.update(status="calibrated", calibration_evidence=["policy unit input"],
                           profile_validation_evidence=["policy unit input"], model_lock_sha256="unit-lock")
        self.policy["minimums"] = dict(subject_similarity=0., style_margin=0., quality_subject=50.)
        self.policy["maximums"] = dict(border_fraction=.1, secondary_component_fraction=.1)

    def row(self, seed, quality, facing="right", view="side"):
        # These numerical values exercise policy logic; they are not model predictions.
        pair = profile.assess(observation(facing, view), observation("left" if facing == "right" else "right", view))
        pair.update(image_sha256=f"image-{seed}", mask_sha256=f"mask-{seed}")
        return {"seed": seed, "image_sha256": f"image-{seed}", "mask_sha256": f"mask-{seed}", "profile": pair,
                "geometry": {"mask_nonempty": True, "border_fraction": 0., "secondary_component_fraction": 0.},
                "metrics": {"subject_similarity": .3, "style_margin": .1, "quality_subject": quality},
                "instrument": {"original_above_strong_blur": True}}

    def decide(self, rows, policy=None):
        return score.decide(rows, self.cfg, policy or self.policy, "unit-lock")

    def test_left_and_oblique_cannot_win_with_higher_quality(self):
        result = self.decide([self.row(0, 95., "left"), self.row(1, 90., view="oblique"), self.row(2, 70.)])
        self.assertEqual(result["selected_seed"], 2)
        self.assertEqual(result["facing_estimate"], "right")
        self.assertFalse(result["automatic_reconstruction_authorized"])

    def test_clip_facing_cannot_influence_v2_selection(self):
        row = self.row(0, 80.)
        row["metrics"].update(facing="left", facing_margin=float("nan"), profile_margin=-1., flip_consistent=False)
        self.assertEqual(self.decide([row])["selected_seed"], 0)

    def test_uncertain_profile_stops_without_regeneration(self):
        row = self.row(0, 80.)
        row["profile"].update(status="uncertain", reason="mirror_disagreement")
        result = self.decide([row])
        self.assertEqual(result["status"], "profile_check_required")
        self.assertEqual(score.retry_action(result["status"], 0, 2), "stop")
        del row["profile"]
        self.assertEqual(self.decide([row])["status"], "profile_check_required")

    def test_known_wrong_profile_can_trigger_bounded_retry(self):
        result = self.decide([self.row(0, 80., "left")])
        self.assertEqual(result["status"], "none_passed")
        self.assertEqual(score.retry_action(result["status"], 0, 2), "regenerate")

    def test_refused_mask_does_not_wait_for_a_profile_observation(self):
        row = self.row(0, 80.)
        row["geometry"]["mask_nonempty"] = False
        row["profile"].update(status="uncertain", reason="mask_refused")
        result = self.decide([row])
        self.assertEqual(result["status"], "none_passed")
        self.assertNotIn("profile_unresolved", result["candidates"][0]["reasons"])

    def test_profile_pixels_and_validation_evidence_are_required(self):
        row = self.row(0, 80.)
        row["profile"]["image_sha256"] = "different image"
        with self.assertRaisesRegex(RuntimeError, "different candidate"):
            self.decide([row])
        for evidence in ([], "not an evidence list", ["  "]):
            bad = copy.deepcopy(self.policy)
            bad["profile_validation_evidence"] = evidence
            with self.assertRaisesRegex(RuntimeError, "profile-validation"):
                self.decide([self.row(0, 80.)], bad)

    def test_default_policy_measures_without_acceptance(self):
        result = self.decide([self.row(0, 80.)], score.policy_template(self.cfg))
        self.assertEqual(result["status"], "calibration_required")
        self.assertIsNone(result["selected_seed"])

    def test_appearance_score_and_profile_configs_have_distinct_roles(self):
        self.assertNotIn("view_texts", self.cfg)
        self.assertIn("render", self.cfg["style_texts"])
        self.assertEqual(score.minimums(self.cfg), ("subject_similarity", "style_margin", "quality_subject"))
        changed = copy.deepcopy(self.cfg)
        changed["profile_spec"]["prompt"] += " Changed measurement."
        self.assertNotEqual(score.metric_signature(self.cfg), score.metric_signature(changed))


if __name__ == "__main__":
    unittest.main()
