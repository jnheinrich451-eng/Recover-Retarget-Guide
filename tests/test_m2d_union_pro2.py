"""CPU contract checks on returned images/traces; no synthetic model output."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from scripts import probe_m2d_union_pro2 as probe


class Pro2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = probe.ROOT / 'measurements/m2d'
        cls.cfg = probe.configuration()
        cls.assets = cls.root / 'union_pro2_assets_v1'
        if not (cls.root / cls.cfg['source_experiment'] / 'suite_manifest.json').exists() or not cls.assets.exists():
            raise unittest.SkipTest('Returned source results and downloaded official fixture required')
        cls.temp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.temp.name) / 'probe'
        cls.m = probe.prepare(cls.cfg, cls.root, cls.out, cls.assets)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_real_inputs_and_idempotent_preparation(self):
        self.assertEqual(self.m, probe.prepare(self.cfg, self.root, self.out, self.assets))
        self.assertEqual(self.m, probe.verified(self.out, self.cfg))
        for name in ('depth.png', 'combined_edges.png', 'mask.png'):
            relative = 'inputs/284_vehicle/' + name
            self.assertEqual((self.out / relative).read_bytes(),
                             (self.root / self.cfg['source_experiment'] / relative).read_bytes())
        self.assertFalse((self.out / 'generation').exists())
        self.assertFalse(self.m['production_handoff_authorized'])

    def test_only_checkpoint_and_mode_interface_change(self):
        for seed in self.cfg['seeds']:
            _, old = probe.expected_call(self.out, self.m, seed, 'pro1')
            _, new = probe.expected_call(self.out, self.m, seed, 'pro2')
            self.assertEqual({k for k in old if old[k] != new[k]},
                             {'arm', 'modes', 'model_revisions', 'controlnet_model'})
            self.assertEqual(old['modes'], [2, 0])
            self.assertEqual(new['modes'], [None, None])
            self.assertEqual(new['active_steps'], [12, 12])
            self.assertEqual(new['scales'], [.6, .2])
            self.assertEqual(new['prompt'], 'A red car driving on a city street.')
            historical = self.m['baselines'][str(seed)]['call']
            for key in ('model_revisions', 'modes', 'control_pixels_sha256', 'prompt', 'steps', 'scales'):
                self.assertEqual(old[key], historical[key])

    def test_publisher_fixture_preserves_original_grid(self):
        images, call = probe.expected_call(self.out, self.m, 42, 'smoke')
        self.assertEqual(images[0].size, (832, 1280))
        self.assertEqual((call['width'], call['height']), images[0].size)
        self.assertEqual(call['scales'], [.7])
        self.assertEqual(call['control_guidance_end'], [.8])
        self.assertEqual(call['modes'], [None])
        self.assertEqual(call['steps'], 30)
        self.assertIn('serene beach', call['prompt'])

    def real_record(self):
        # Re-express genuine original-Pro records with branch labels. This does
        # not represent or predict a Pro 2.0 output.
        row = deepcopy(self.m['baselines']['4'])
        row['call'].update(width=1024, height=1024, branches=['depth', 'edges'],
                           controlnet_model=self.m['settings']['controlnet_model'])
        row['result']['actual_control_calls'] = {
            'depth': row['result']['actual_union_calls']['2'],
            'edges': row['result']['actual_union_calls']['0']}
        return row

    def test_real_replay_and_corruption_detection(self):
        row = self.real_record()
        self.assertTrue(all(probe.replay_check(row, self.m['baselines']['4']).values()))
        altered = deepcopy(row)
        altered['result']['image_sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'replay changed'):
            probe.replay_check(altered, self.m['baselines']['4'])
        # Replaying the same model cannot masquerade as a Pro 2.0 comparison.
        checks = probe.pair_checks(row, row)
        self.assertFalse(checks['correct_mode_change'])
        self.assertFalse(checks['different_controlnet'])
        self.assertTrue(checks['same_encoded_edges'])
        altered = deepcopy(row)
        altered['result']['actual_control_calls']['edges']['encoded_control_sha256'] = '0' * 64
        self.assertFalse(probe.pair_checks(row, altered)['same_encoded_edges'])
        altered = deepcopy(row)
        altered['call']['initial_latents_sha256'] = '0' * 64
        self.assertFalse(probe.pair_checks(row, altered)['same_initial_latents_sha256'])

    def test_corrupt_inputs_and_missing_results_refused(self):
        path = self.out / 'inputs/284_vehicle/combined_edges.png'
        before = path.read_bytes()
        try:
            path.write_bytes(b'damaged')
            with self.assertRaisesRegex(RuntimeError, 'Changed completed'):
                probe.verified(self.out, self.cfg)
        finally:
            path.write_bytes(before)
        summary = probe.report(self.out, self.m)
        self.assertFalse(summary['complete'])
        self.assertEqual(summary['visual_quality_acceptance'], 'not_measured')
        with self.assertRaisesRegex(RuntimeError, 'Missing comparison'):
            probe.report(self.out, self.m, complete=True)
        with self.assertRaises(RuntimeError):
            probe.guard(self.root, self.root / self.cfg['source_experiment'], self.cfg)


if __name__ == '__main__':
    unittest.main()
