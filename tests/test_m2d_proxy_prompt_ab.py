"""Paired experiment bookkeeping and real returned observation checks; no model mocks."""
from argparse import Namespace
import copy
from pathlib import Path
import tempfile
import unittest

import yaml

from scripts import probe_m2d_proxy_prompt_ab as probe


class PromptABTests(unittest.TestCase):
    def setUp(self):
        self.spec = yaml.safe_load(probe.DEFAULT.read_text(encoding="utf-8"))
        self.source, _, _ = probe.bear.configuration()

    def test_pair_changes_only_prompt_and_uses_fresh_matched_seeds(self):
        configs = probe.arm_configs(self.source, self.spec)
        a, b = [configs[k] for k in probe.ARMS]
        self.assertEqual(a["proxy_prompt"], self.source["proxy_prompt"])
        self.assertEqual(b["proxy_prompt"], a["proxy_prompt"] + ". " + self.spec["spatial_suffix"])
        self.assertEqual(a["proxy_seeds"], [4, 5, 6, 7])
        self.assertEqual({k for k in a if a[k] != b[k]}, {"proxy_prompt"})
        self.assertEqual({k for k in a if a[k] != self.source[k]}, {"proxy_seeds"})

    def test_reused_or_ambiguous_seeds_refused(self):
        for seeds in ([0, 1, 2, 3], [4, 4, 5, 6], [4, 5], [-1, 4, 5, 6], [True, 4, 5, 6]):
            with self.subTest(seeds=seeds), self.assertRaises(RuntimeError):
                probe.arm_configs(self.source, {**self.spec, "seeds": seeds})

    def test_wrong_baseline_or_empty_treatment_refused(self):
        with self.assertRaises(RuntimeError):
            probe.arm_configs({**self.source, "proxy_prompt": "a bear"}, self.spec)
        with self.assertRaises(RuntimeError):
            probe.arm_configs(self.source, {**self.spec, "spatial_suffix": " "})

    def test_preparation_on_returned_metadata_is_resumable_and_frozen(self):
        root = probe.ROOT / "measurements/m2d"
        if not (root / self.spec["source_scores"] / "models.json").exists():
            self.skipTest("Returned Colab metadata is not in this checkout")
        with tempfile.TemporaryDirectory() as temp:
            args = Namespace(config=probe.DEFAULT, probe_root=root, drive_root=Path(temp) / "drive",
                             output_dir=Path(temp) / "ab")
            identity = probe.prepare(args)
            before = {p.relative_to(args.output_dir): p.read_bytes() for p in args.output_dir.rglob("*") if p.is_file()}
            self.assertEqual(identity, probe.prepare(args))
            self.assertEqual(before, {p.relative_to(args.output_dir): p.read_bytes() for p in args.output_dir.rglob("*") if p.is_file()})
            self.assertEqual(identity["generation_models"]["black-forest-labs/FLUX.1-dev"],
                             "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21")
            changed = copy.deepcopy(self.spec)
            changed["spatial_suffix"] += " changed"
            args.config = Path(temp) / "different.yaml"
            args.config.write_text(yaml.safe_dump(changed), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Existing artifact differs"):
                probe.prepare(args)

    def test_returned_observer_disagreement_is_not_accepted(self):
        path = probe.ROOT / "measurements/m2d/035_bear_right_profile_v3/scores.json"
        if not path.exists():
            self.skipTest("Returned A100 observations are not in this checkout")
        rows = probe.core.read_json(path)
        diagnostics = {r["seed"]: probe.diagnostic_row(probe.ARMS[0], r) for r in rows}
        self.assertEqual(diagnostics[0]["requested_direction_status"], "mismatch")
        self.assertEqual(diagnostics[1]["combined_profile"], "supported")
        for seed in (2, 3):
            self.assertEqual(diagnostics[seed]["combined_profile"], "uncertain")
            self.assertEqual(diagnostics[seed]["profile_reason"], "mirror_disagreement")
        self.assertEqual(len({r["candidate_id"] for arm in probe.ARMS for r in
                              (probe.diagnostic_row(arm, row) for row in rows)}), 8)


if __name__ == "__main__":
    unittest.main()
