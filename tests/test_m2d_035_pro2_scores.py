"""Saved-image scoring contracts; real returned data, no model mocks."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from scripts import score_m2d_035_pro2 as probe


class SavedPro2ScoresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = probe.ROOT / 'measurements/m2d'
        cls.cfg = probe.configuration()
        if not (cls.root / cls.cfg['source_experiment'] / 'generation/bear/seed_4/guidance_18/result.json').exists():
            raise unittest.SkipTest('Returned 18-step images required')
        cls.temp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.temp.name) / 'scores'
        cls.m = probe.prepare(cls.cfg, cls.root, cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_exact_four_candidates_and_idempotent_preparation(self):
        self.assertEqual(self.m, probe.prepare(self.cfg, self.root, self.out))
        self.assertEqual(self.m, probe.verified(self.out, self.cfg))
        self.assertEqual(len(list((self.out / 'candidates').rglob('*.png'))), 4)
        for subject in self.cfg['subjects']:
            for seed in self.cfg['seeds']:
                old = self.root / self.cfg['source_experiment'] / f'generation/{subject}/seed_{seed}/guidance_18/image.png'
                self.assertEqual((self.out / f'candidates/{subject}/seed_{seed}.png').read_bytes(), old.read_bytes())
        self.assertFalse((self.out / 'generation').exists())
        self.assertFalse((self.out / 'measurement').exists())

    def test_subject_specific_geometry_and_identity_and_shared_reference(self):
        bear, car = [self.m['cases'][s] for s in ('bear', 'car')]
        self.assertEqual(bear['appearance']['subject_text'], 'a brown bear')
        self.assertEqual(car['appearance']['subject_text'], 'a red car')
        self.assertEqual(car['association']['subject_query'], 'a red car')
        self.assertEqual(bear['geometry']['rendered_area'], 35000)
        self.assertEqual(car['geometry']['rendered_area'], 43528)
        self.assertEqual(bear['geometry']['centroid_xy'], car['geometry']['centroid_xy'])
        self.assertEqual(bear['geometry']['K_analysis'], car['geometry']['K_analysis'])
        self.assertEqual(bear['traces'], car['traces'])
        self.assertNotEqual(self.m['fixtures']['car']['geometry'], car['geometry'])

    def test_known_fixtures_match_archived_real_segmentations(self):
        root = self.root / self.cfg['scoring_source']
        for subject, case in self.cfg['fixture_cases'].items():
            record = probe.core.read_json(root / f'measurement/{case}/fixture/result.json')
            self.assertTrue(probe.fixture_ok(record))
            spec = self.m['fixtures'][subject]['fixture']
            folder = root / f'measurement/{case}/fixture'
            self.assertTrue(probe.original.known_points(probe.prior.mask(folder / 'road.png'), spec['road'])['passed'])
            self.assertTrue(probe.association.check_fixture(probe.prior.mask(folder / 'subject.png'),
                            record['association'], spec['subject'])['passed'])
        jobs = list(probe.tasks(self.out, self.m))
        self.assertEqual(len(jobs), 6)
        self.assertEqual([r[1] for r in jobs], [None, None, 4, 6, 4, 6])

    def test_unchanged_decision_on_real_old_bear_scores(self):
        # Replay old observations without pretending they belong to the new images.
        root = self.root / '035_bear_anchor_seed_scores_v1'
        rows = [probe.core.read_json(root / f'measurement/seed_{s}/result.json') for s in range(4)]
        fixture = probe.core.read_json(root / 'measurement/fixture/result.json')
        before = probe.scoring.decision(rows, fixture, self.m)
        after = probe.decision(rows, fixture, self.m)
        self.assertEqual(before['candidates'], after['candidates'])
        self.assertEqual(after['selected_seed'], 1)
        self.assertEqual(after['policy'], probe.previous.tie.POLICY)
        bad = deepcopy(fixture)
        bad['selftest']['road']['passed'] = False
        self.assertIsNone(probe.decision(rows, bad, self.m)['selected_seed'])

    def test_no_selection_without_measurements_and_corruption_refused(self):
        result = probe.report(self.out, self.m)
        self.assertFalse(result['complete'])
        self.assertFalse((self.out / 'scores.csv').exists())
        with self.assertRaisesRegex(RuntimeError, 'Missing/mislabelled'):
            probe.select(self.out, self.m)
        path = self.out / 'candidates/bear/seed_4.png'
        before = path.read_bytes()
        try:
            path.write_bytes(b'damaged')
            with self.assertRaisesRegex(RuntimeError, 'Changed completed'):
                probe.verified(self.out, self.cfg)
        finally:
            path.write_bytes(before)
        with self.assertRaises(RuntimeError):
            probe.guard(self.root, self.root / self.cfg['source_experiment'], self.cfg)


if __name__ == '__main__':
    unittest.main()
