"""035 preparation and pair audits use returned controls and real inference."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from scripts import probe_m2d_035_union_pro2 as probe


class CliffPro2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = probe.ROOT / 'measurements/m2d'
        cls.cfg = probe.configuration()
        if not (cls.root / cls.cfg['source_recipe'] / 'smoke/result.json').exists():
            raise unittest.SkipTest('Returned Pro 2.0 results required')
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name) / 'cliff'
        cls.m = probe.prepare(cls.cfg, cls.root, cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_verified_source_and_byte_identical_controls(self):
        self.assertEqual(self.m, probe.prepare(self.cfg, self.root, self.output))
        self.assertEqual(self.m, probe.verified(self.output, self.cfg))
        for subject in ('car', 'bear'):
            for name in ('depth', 'edges', 'mask'):
                path = f'inputs/{subject}/{name}.png'
                self.assertEqual((self.output / path).read_bytes(),
                                 (self.root / self.cfg['source_controls'] / path).read_bytes())
        self.assertFalse((self.output / 'generation').exists())
        self.assertEqual(self.m['geometry']['K_analysis'],
                         probe.core.read_json(self.root / self.cfg['source_geometry'])['K_1024'])

    def test_eight_calls_fixed_recipe_and_simple_prompts(self):
        rows, pairs = probe.records(self.output, self.m)
        self.assertEqual(len(rows), 8)
        self.assertFalse(pairs)
        template = self.m['template_call']
        for subject in ('car', 'bear'):
            for seed in (4, 6):
                _, a = probe.expected_call(self.output, self.m, subject, seed, 'pro1')
                _, b = probe.expected_call(self.output, self.m, subject, seed, 'pro2')
                self.assertEqual({k for k in a if a[k] != b[k]},
                                 {'arm', 'modes', 'model_revisions', 'controlnet_model'})
                self.assertEqual(b['prompt'], self.cfg['prompts'][subject])
                self.assertEqual(b['control_vae_rng_seed'], seed)
                self.assertEqual(b['modes'], [None, None])
                for key in ('width', 'height', 'steps', 'scales', 'active_steps', 'guidance',
                            'max_sequence_length', 'control_guidance_start', 'control_guidance_end'):
                    self.assertEqual(b[key], template[key])
        with self.assertRaisesRegex(RuntimeError, 'Missing 035'):
            probe.records(self.output, self.m, complete=True)

    def test_distinct_subject_controls_and_historical_context(self):
        _, car = probe.expected_call(self.output, self.m, 'car', 4, 'pro2')
        _, bear = probe.expected_call(self.output, self.m, 'bear', 4, 'pro2')
        self.assertNotEqual(car['control_pixels_sha256'], bear['control_pixels_sha256'])
        self.assertNotEqual(car['prompt'], bear['prompt'])
        for subject in ('car', 'bear'):
            old = self.m['historical'][subject]['call']
            self.assertEqual(old['seed'], 1)
            self.assertEqual(old['active_steps'], [24, 24])
        summary = probe.report(self.output, self.m)
        self.assertFalse(summary['complete'])
        self.assertEqual(summary['automatic_selection'], 'not_run')
        page = (self.output / 'review.html').read_text()
        self.assertIn('different recipe/prompt/seed', page)

    def test_actual_pro2_pair_and_corrupt_metadata(self):
        root = self.root / self.cfg['source_recipe']
        old = probe.verified_recipe(root)
        pair = [probe.previous.read_record(root, old, arm, 4) for arm in ('pro1', 'pro2')]
        self.assertTrue(all(probe.previous.pair_checks(*pair).values()))
        bad = deepcopy(pair[1])
        bad['call']['initial_latents_sha256'] = '0' * 64
        self.assertFalse(probe.previous.pair_checks(pair[0], bad)['same_initial_latents_sha256'])
        bad = deepcopy(pair[1])
        bad['result']['actual_control_calls']['edges']['encoded_control_sha256'] = '0' * 64
        self.assertFalse(probe.previous.pair_checks(pair[0], bad)['same_encoded_edges'])

    def test_corrupt_artifact_and_unsafe_output_refused(self):
        path = self.output / 'inputs/car/edges.png'
        before = path.read_bytes()
        try:
            path.write_bytes(b'damaged')
            with self.assertRaisesRegex(RuntimeError, 'Changed completed'):
                probe.verified(self.output, self.cfg)
        finally:
            path.write_bytes(before)
        with self.assertRaises(RuntimeError):
            probe.guard(self.root, self.root / self.cfg['source_controls'], self.cfg)


if __name__ == '__main__':
    unittest.main()
