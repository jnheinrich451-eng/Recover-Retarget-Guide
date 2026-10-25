"""Style parsing/decision contracts and preparation of actual saved pixels.

JSON examples test decision rules only; no external model is mocked and no
fixture verdict is saved as real inference or calibrated style evidence.
"""
from argparse import Namespace
import copy
import json
from pathlib import Path
import tempfile
import unittest

import yaml
from scripts import probe_m2d_proxy_style as style


def answer(label):
    return {"observation": {"appearance": label, "evidence": "Parser/decision fixture only."}}


def pair(full="photographic", crop=None):
    return {"full": answer(full), "subject_crop": answer(full if crop is None else crop)}


class StyleContracts(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load(style.DEFAULT.read_text(encoding="utf-8"))

    def attempt(self, raw, retry=False):
        parsed, error = style.parse(raw)
        prompt = self.cfg["prompt"] + ("\n\n" + self.cfg["format_retry_prompt"] if retry else "")
        return {"raw": raw, "observation": parsed, "parse_error": error,
                "prompt_sha256": style.core.digest(prompt.encode())}

    def test_labels_and_code_fence(self):
        for label in style.LABELS:
            raw = json.dumps(answer(label)["observation"])
            self.assertEqual(style.parse(raw), (answer(label)["observation"], None))
            self.assertEqual(style.parse("```json\n" + raw + "\n```"), style.parse(raw))

    def test_invalid_does_not_become_photo(self):
        for raw in ("photo", "{}", "[]", '{"appearance": "photographic", "evidence": ""}',
                    '{"appearance": [], "evidence": "x"}',
                    '{"appearance": "photographic", "evidence": "x", "confidence": 1}',
                    '{"appearance": "real", "evidence": "x"}'):
            self.assertEqual(style.parse(raw), (None, "invalid_structured_answer"))

    def test_nonphoto_and_uncertain_not_retried(self):
        for label in style.LABELS:
            attempt = self.attempt(json.dumps(answer(label)["observation"]))
            self.assertEqual(style.finish([attempt], self.cfg)["observation"], answer(label)["observation"])
            retry = self.attempt(json.dumps(answer("photographic")["observation"]), retry=True)
            with self.assertRaisesRegex(RuntimeError, "Cannot retry"):
                style.finish([attempt, retry], self.cfg)

    def test_invalid_format_may_retry_once(self):
        invalid = self.attempt("not JSON")
        with self.assertRaisesRegex(RuntimeError, "unfinished"):
            style.finish([invalid], self.cfg)
        retry = self.attempt(json.dumps(answer("cartoon")["observation"]), retry=True)
        self.assertEqual(style.finish([invalid, retry], self.cfg)["observation"]["appearance"], "cartoon")

    def test_response_must_match_raw_and_question(self):
        raw = json.dumps(answer("cartoon")["observation"])
        item = self.attempt(raw)
        item["observation"] = answer("photographic")["observation"]
        with self.assertRaisesRegex(RuntimeError, "raw output"):
            style.finish([item], self.cfg)
        item = self.attempt(raw)
        item["prompt_sha256"] = "changed"
        with self.assertRaisesRegex(RuntimeError, "question changed"):
            style.finish([item], self.cfg)

    def test_two_photo_views_required(self):
        self.assertEqual(style.assess(pair())["status"], "supported")
        for label in ("cartoon", "stylized_render", "toy"):
            self.assertEqual(style.assess(pair(label))["status"], "rejected")
            self.assertEqual(style.assess(pair("photographic", label))["status"], "uncertain")
        for inputs in ({}, {"full": answer("photographic")}, pair("uncertain")):
            self.assertEqual(style.assess(inputs)["status"], "uncertain")


class ReturnedInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = style.ROOT / "measurements/m2d"
        cls.source = cls.root / "035_bear_prompt_ab_v1/b_spatial"
        if not (cls.source / "scoring/score_index.json").exists():
            raise unittest.SkipTest("Returned candidate evidence absent")
        cls.evidence = style.facing.load_evidence(cls.source / "candidates", cls.source / "scoring")
        cls.base = style.facing.decide(cls.evidence)

    def test_cartoon_cannot_win_by_quality(self):
        base = copy.deepcopy(self.base)
        next(r for r in base["candidates"] if r["seed"] == 5)["metrics"]["quality_subject"] = 999
        # Artificial labels exercise the rule; these are NOT Qwen results.
        observations = {r["seed"]: pair() for r in base["candidates"]}
        observations[5] = pair("cartoon")
        out = style.shortlist(base, observations)
        self.assertNotIn(5, out["ranked_seeds"])
        self.assertEqual(out["provisional_seed"], 6)
        self.assertFalse(out["automatic_reconstruction_authorized"])

    def test_photographic_left_can_win_without_direction_quota(self):
        observations = {r["seed"]: pair("cartoon") for r in self.base["candidates"]}
        observations[5] = pair()
        out = style.shortlist(self.base, observations)
        self.assertEqual((out["provisional_seed"], out["provisional_facing"]), (5, "left"))
        self.assertEqual(out["ranked_seeds"], [5])

    def test_no_qualifying_candidate_has_no_winner(self):
        for label in ("cartoon", "uncertain"):
            observations = {r["seed"]: pair(label) for r in self.base["candidates"]}
            out = style.shortlist(self.base, observations)
            self.assertIsNone(out["provisional_seed"])
            self.assertEqual(out["ranked_seeds"], [])

    def test_style_does_not_override_geometry_failure(self):
        base = copy.deepcopy(self.base)
        for row in base["candidates"]:
            row["geometry_observation_supported"] = False
        out = style.shortlist(base, {r["seed"]: pair() for r in base["candidates"]})
        self.assertIsNone(out["provisional_seed"])

    def test_census_required(self):
        with self.assertRaisesRegex(RuntimeError, "census"):
            style.shortlist(self.base, {})

    def test_actual_prepare_resumes_and_preserves_source_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = Namespace(config=style.DEFAULT, probe_root=self.root, output_dir=Path(tmp) / "style",
                             device="cpu", gpu_memory_gib=None)
            cfg, base = style.prepare(args)
            style.prepare(args)
            for row in base["candidates"]:
                path = args.output_dir / f"seed_{row['seed']}"
                self.assertEqual(style.core.file_hash(path / "full.png"), row["image_sha256"])
                inp = style.core.read_json(path / "inputs.json")
                self.assertIsNotNone(inp["crop_xyxy"])
            self.assertFalse((args.output_dir / "selection.json").exists())
            (args.output_dir / "seed_5/full.png").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "Existing artifact differs"):
                style.prepare(args)


if __name__ == "__main__":
    unittest.main()
