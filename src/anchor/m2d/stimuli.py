"""Synthetic polarity stimuli (SKILL §2), composed DIRECTLY as disparity fields with C3's own
functions — polarity is a fact about the generator's preprocessor, not about any clip, so the stimuli
are purpose-built and maximally decisive. Each makes inversion VISIBLE:

  S1 ground plane receding to a horizon + one box     inverted: the plane reads as a ceiling
  S2 a near bar CROSSING a far panel                   inverted: the occlusion order flips (topological
                                                       readout — no estimator, no depth judgement)
  S3 a corridor (floor, ceiling, two walls)            inverted: the vanishing point comes forward

All at 1024 x 1024 under K = [[1024,0,512],[0,1024,512],[0,0,1]], camera 1.5 above a level ground.
"""
from __future__ import annotations

import numpy as np

from ..m2c import compose as _c3
from ..m2c import proxy as _proxy
from ..m2c import sanitise as _san

PX = 1024
K = np.array([[1024.0, 0, 512.0], [0, 1024.0, 512.0], [0, 0, 1.0]])
N_UP = np.array([0.0, -1.0, 0.0])
CAM_H = 1.5


def _ground():
    z = _san.plane_disparity(N_UP, CAM_H, K, PX, PX)
    valid = z > 0                                   # above the horizon: sky, zeta = 0
    return np.where(valid, z, 0.0), valid


def _box(half, centre):
    """Axis-aligned box placed in camera coordinates; returns (zeta_obj, M_obj)."""
    b = _proxy.box(half=half)
    zb, mk = _c3.render_object(b.V, b.F, np.asarray(centre, float), 1.0, np.eye(3), K, PX, PX)
    return _c3.object_disparity(zb, mk), mk


def _finish(name, zeta, valid, regions, ordering, p_low=0.02, p_high=0.98):
    D, ql, qh = _c3.normalise(zeta, valid, p_low, p_high)
    D32 = D.astype(np.float32)
    return {"name": name, "zeta_ctrl": zeta, "valid": valid, "D_ctrl": D32, "D_ctrl_inv": (np.float32(1) - D32).astype(np.float32),
            "q_low": ql, "q_high": qh, "regions": regions, "known_ordering": ordering}


def s1_ground_box():
    zg, vg = _ground()
    zo, mo = _box((0.8, 0.8, 0.8), (0.0, CAM_H - 0.8, 6.0))                 # box standing on the ground at z = 6
    zeta = np.maximum(zg, zo)
    return _finish("S1_ground_box", zeta, vg | mo, {"object": mo, "ground": vg & ~mo},
                   "the ground recedes AWAY toward the horizon (near at the bottom of the image); the box stands on it")


def s2_crossing_occluder():
    zg, vg = _ground()
    zf, mf = _box((3.0, 1.0, 0.3), (0.0, CAM_H - 1.0, 10.0))                # far: a wide panel at z = 10
    zn, mn = _box((0.35, 1.3, 0.35), (0.0, CAM_H - 1.3, 4.0))               # near: a tall bar at z = 4, crossing the panel's middle
    zeta = np.maximum(np.maximum(zg, zf), zn)
    overlap = mf & mn
    assert overlap.sum() > 2000, "the bar must cross the panel"
    assert (zn[overlap] > zf[overlap]).all(), "the bar is nearer than the panel wherever they overlap"
    return _finish("S2_crossing_occluder", zeta, vg | mf | mn, {"near_bar": mn, "far_panel": mf & ~mn, "overlap": overlap},
                   "the tall narrow bar is IN FRONT of the wide panel: the bar crosses over the panel, the panel never covers the bar")


def s3_corridor():
    """Floor y = +1.5, ceiling y = -1.5, walls x = +-2.0: the visible surface per pixel is the nearest
    plane, i.e. the largest positive disparity; the far end is the vanishing point (zeta -> 0)."""
    planes = [(np.array([0.0, -1.0, 0.0]), 1.5), (np.array([0.0, 1.0, 0.0]), 1.5),
              (np.array([-1.0, 0.0, 0.0]), 2.0), (np.array([1.0, 0.0, 0.0]), 2.0)]
    z = np.zeros((PX, PX))
    for n, d in planes:
        z = np.maximum(z, _san.plane_disparity(n, d, K, PX, PX))
    valid = z > 0
    return _finish("S3_corridor", z, valid, {"surfaces": valid},
                   "a corridor receding AWAY into the image: walls, floor and ceiling converge toward a far end at the centre")


def all_synthetic():
    return [s1_ground_box(), s2_crossing_occluder(), s3_corridor()]


def to_control_uint8(D):
    """The exact 3-channel uint8 image handed to the generator: round(D * 255), no transfer curve."""
    g = (np.clip(np.asarray(D, np.float32), 0, 1) * 255.0 + 0.5).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=2)
