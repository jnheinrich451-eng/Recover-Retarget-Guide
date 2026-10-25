"""CPU geometric fixtures and artifact checks. No external models are mocked."""
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import probe_m2d_car_alignment as P
from src.anchor.m2c import proxy


class AlignmentProbeTests(unittest.TestCase):
    def test_refuses_overlapping_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source'
            for dest in (source, source/'child', source.parent):
                with self.assertRaises(RuntimeError):
                    P.validate_output(dest, [source])
            P.validate_output(Path(tmp)/'separate', [source])

    def test_binary_mask_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'mask.png'
            a = np.zeros((20, 30), np.uint8); a[4:15, 5:20] = 255
            Image.fromarray(a).save(path)
            self.assertEqual(P.read_mask(path, a.shape).sum(), 165)
            with self.assertRaises(RuntimeError): P.read_mask(path, (30, 20))
            a[0,0] = 127; Image.fromarray(a).save(path)
            with self.assertRaises(RuntimeError): P.read_mask(path, a.shape)
            Image.fromarray(np.zeros_like(a)).save(path)
            with self.assertRaises(RuntimeError): P.read_mask(path, a.shape)

    def test_renderer_cache_is_exact_and_restored_on_error(self):
        m = proxy.box()
        values = (m.V, m.F, np.array([0.,0,4]), .5, np.eye(3),
                  np.array([[80.,0,32],[0,90.,32],[0,0,1]]), 64, 64)
        original = P.compose.render_object
        expected = original(*values)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'deliberate'):
                with P.cached_renderer(folder):
                    for _ in range(2):
                        actual = P.compose.render_object(*values)
                        for x,y in zip(actual,expected): np.testing.assert_array_equal(x,y)
                    raise ValueError('deliberate')
            self.assertIs(P.compose.render_object, original)
            next(folder.glob('*.npz')).write_bytes(b'corrupt cache')
            with P.cached_renderer(folder):
                with self.assertRaisesRegex(RuntimeError, 'checksum'):
                    P.compose.render_object(*values)

    def test_geometric_scan_sign_resume_and_mask_conflict(self):
        # Two analytic boxes with an offset bump break front/back symmetry.
        # This is an exact geometric calibration target, not a model stand-in.
        body = proxy.box((.9, .35, .4)); bump = proxy.box((.25, .22, .2))
        V = np.concatenate([body.V, bump.V + [.8, -.12, .25]])
        F = np.concatenate([body.F, bump.F + len(body.V)])
        c = dict(V=V, F=F, X=np.array([.05, .2, 4.]), s=.9,
                 R=P.D.rot_about([0,-1,0], 25), n=np.array([0.,-1,0]),
                 K=np.array([[150.,0,64],[0,190,64],[0,0,1]]), size=128,
                 cfg={'self_test_yaw_deg':12., 'self_test_min_tolerance_deg':.5},
                 scan_kw=dict(scan_deg=45., step_deg=1., tol=.01, refine=True))
        c['M'] = P.render(c, 0)
        truth = P.render(c, -18.)
        c['images'] = {'geometric_fixture': np.repeat((truth*180).astype(np.uint8)[...,None],3,axis=2)}
        with tempfile.TemporaryDirectory() as tmp:
            out, masks = Path(tmp)/'out', Path(tmp)/'masks'
            masks.mkdir()
            Image.fromarray(truth.astype(np.uint8)*255).save(masks/'geometric_fixture.png')
            args = SimpleNamespace(output_dir=out, mask_dir=masks)
            P.save_image(out/'geometric_fixture/image.png', c['images']['geometric_fixture'])
            P.make_masks(args, c)
            P.scan(args, c)
            r = P.read_json(out/'geometric_fixture/result.json')
            self.assertLess(abs(r['delta_deg']+18), max(.5, r['curve_width_deg']/2))
            self.assertAlmostEqual(r['iou_at_zero'], P.D.iou(c['M'],truth))
            self.assertGreater(r['front_back_margin'], 0)
            self.assertTrue(r['status'].startswith('PROVISIONAL'))
            before = {p.relative_to(out):p.read_bytes() for p in out.rglob('*') if p.is_file()}
            P.make_masks(args, c); P.scan(args, c)
            self.assertEqual(before, {p.relative_to(out):p.read_bytes() for p in out.rglob('*') if p.is_file()})
            P.feedback(out)
            self.assertTrue((out/'car_alignment_feedback.zip').is_file())
            # An edited mask cannot silently reuse previously computed curves.
            Image.fromarray(np.roll(truth, 1, axis=0).astype(np.uint8)*255).save(masks/'geometric_fixture.png')
            with self.assertRaises(RuntimeError): P.make_masks(args, c)


if __name__ == '__main__':
    unittest.main()
