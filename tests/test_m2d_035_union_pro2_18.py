"""Guidance comparison contracts using real returned controls and traces."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from scripts import probe_m2d_035_union_pro2_18 as probe


class Guidance18Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = probe.ROOT / 'measurements/m2d'
        cls.cfg = probe.configuration()
        if not (cls.root / cls.cfg['source_experiment'] / 'generation/bear/seed_4/pro2/result.json').is_file():
            raise unittest.SkipTest('Returned 035 Pro 2.0 results required')
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name) / 'guidance18'
        cls.m = probe.prepare(cls.cfg, cls.root, cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_preserve_real_controls_geometry_and_resumable_prepare(self):
        self.assertEqual(self.m, probe.prepare(self.cfg, self.root, self.output))
        self.assertEqual(self.m, probe.verified(self.output, self.cfg))
        source = self.root / self.cfg['source_experiment']
        for subject in self.cfg['subjects']:
            for name in ('depth', 'edges', 'mask'):
                path = f'inputs/{subject}/{name}.png'
                self.assertEqual((self.output / path).read_bytes(), (source / path).read_bytes())
        self.assertEqual(self.m['geometry'], probe.core.read_json(source / 'suite_manifest.json')['geometry'])
        self.assertFalse((self.output / 'generation').exists())

    def test_only_registered_intervention_changes_and_total_steps_stay_24(self):
        for subject in self.cfg['subjects']:
            for seed in self.cfg['seeds']:
                _, baseline = probe.expected_call(self.output, self.m, subject, seed, 'baseline_12')
                _, trial = probe.expected_call(self.output, self.m, subject, seed, 'guidance_18')
                self.assertEqual({k for k in baseline if baseline[k] != trial[k]}, probe.CHANGED_FIELDS)
                self.assertEqual(trial['steps'], 24)
                self.assertEqual(trial['active_steps'], [18, 18])
                self.assertEqual(trial['control_guidance_end'], [.75, .75])
                self.assertEqual(trial['scales'], [.6, .4])
                archived = self.m['baselines'][subject][str(seed)]['call']
                self.assertEqual({k for k in baseline if baseline[k] != archived[k]}, {'arm'})
                self.assertEqual(trial['prompt'], archived['prompt'])
                self.assertEqual(trial['model_revisions'], archived['model_revisions'])
                self.assertEqual(trial['control_vae_rng_seed'], seed)

    def test_actual_trace_cannot_masquerade_as_18_step_output(self):
        row = self.m['baselines']['bear']['4']
        probe.engine.check_trace(row['call'], row['result'])
        _, trial = probe.expected_call(self.output, self.m, 'bear', 4, 'guidance_18')
        with self.assertRaisesRegex(RuntimeError, 'Wrong control schedule'):
            probe.engine.check_trace(trial, row['result'])

    def test_actual_record_audit_catches_noise_encoding_and_replay_drift(self):
        row = self.m['baselines']['bear']['4']
        self.assertTrue(all(probe.replay_check(row, row).values()))
        for location, key, check in (
            ('call', 'initial_latents_sha256', 'same_initial_latents_sha256'),
            ('call', 'prompt', 'same_prompt'),
            ('call', 'model_revisions', 'same_model_revisions'),
        ):
            corrupt = deepcopy(row)
            corrupt[location][key] = 'changed'
            self.assertFalse(probe.common_checks(row, corrupt)[check])
        corrupt = deepcopy(row)
        corrupt['result']['actual_control_calls']['edges']['encoded_control_sha256'] = '0' * 64
        self.assertFalse(probe.common_checks(row, corrupt)['same_encoded_edges'])
        corrupt = deepcopy(row)
        corrupt['result']['image_sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'replay changed'):
            probe.replay_check(corrupt, row)

    def test_pending_report_and_corrupt_artifact_refusal(self):
        summary = probe.report(self.output, self.m)
        self.assertFalse(summary['complete'])
        self.assertEqual(summary['automatic_selection'], 'not_run')
        self.assertEqual(summary['visual_quality_acceptance'], 'not_measured')
        self.assertFalse(summary['production_handoff_authorized'])
        with self.assertRaisesRegex(RuntimeError, 'Missing guidance'):
            probe.records(self.output, self.m, complete=True)
        path = self.output / 'inputs/bear/edges.png'
        before = path.read_bytes()
        try:
            path.write_bytes(b'damaged')
            with self.assertRaisesRegex(RuntimeError, 'Changed completed'):
                probe.verified(self.output, self.cfg)
        finally:
            path.write_bytes(before)
        with self.assertRaises(RuntimeError):
            probe.guard(self.root, self.root / self.cfg['source_experiment'], self.cfg)


if __name__ == '__main__':
    unittest.main()
