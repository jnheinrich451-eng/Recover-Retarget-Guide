"""E1 composition for SAM 3D Objects (SKILL §3), measured 2026-09-13/14 and accepted by the human.

    X_cv = ((V_local * s) @ R(q_wxyz) + t) @ diag(-1, -1, 1)

The vendor composes with PyTorch3D's `compose_transform(scale, R, t)`, which acts on ROW
vectors (`X @ R`), then the PyTorch3D->OpenCV camera conversion. Measured facts encoded as
asserts, not comments:
  * quaternion order is wxyz (xyzw drops ellipsoid_0 IoU 0.744 -> 0.547);
  * the camera conversion is mandatory (real anchors 0.795 -> 0.000 without it);
  * a column-vector reader must use R^T (it is the same map);
  * `scale` is returned per axis (1,3) and was isotropic in every measured call (P1 10,
    P2 17, P4 12 samples, axis spread 0.0) -> gate: read as a scalar, refuse otherwise.
The pose head's DECOMPOSITION (R alone) is never used downstream (E2 failed: 30 deg seed
spread on a fixed image); only this composition is.
"""
from __future__ import annotations

import numpy as np

P3D_TO_CV = np.diag([-1.0, -1.0, 1.0])       # symmetric, self-inverse; row- and column-form identical


class ScaleAnisotropic(ValueError):
    """The vendor returned a per-axis scale whose axes differ beyond the isotropy tolerance."""


def quat_wxyz_to_matrix(q) -> np.ndarray:
    """numpy port of pytorch3d.transforms.quaternion_to_matrix (real part FIRST)."""
    w, x, y, z = np.asarray(q, np.float64).reshape(4)
    n = w * w + x * x + y * y + z * z
    assert n > 0, "zero quaternion"
    two_s = 2.0 / n
    return np.array([[1 - two_s * (y * y + z * z), two_s * (x * y - z * w), two_s * (x * z + y * w)],
                     [two_s * (x * y + z * w), 1 - two_s * (x * x + z * z), two_s * (y * z - x * w)],
                     [two_s * (x * z - y * w), two_s * (y * z + x * w), 1 - two_s * (x * x + y * y)]])


def read_scale(scale, rel_tol: float) -> float:
    """SKILL §3 gate: per-axis (1,3) scale -> scalar iff max-min < rel_tol * mean."""
    s = np.asarray(scale, np.float64).reshape(-1)
    assert s.shape == (3,) or s.shape == (1,), f"scale must have 1 or 3 entries, got {s.shape}"
    assert np.isfinite(s).all() and (s > 0).all(), f"scale must be finite and positive: {s}"
    if s.shape == (1,):
        return float(s[0])
    spread = float(s.max() - s.min())
    if spread >= rel_tol * float(s.mean()):
        raise ScaleAnisotropic(f"per-axis scale {s.tolist()} is anisotropic (spread {spread:.3g} >= "
                               f"{rel_tol:g} x mean {s.mean():.3g}); the measured convention is isotropic — refusing")
    return float(s.mean())


def compose_objects(V_local, q_wxyz, t, scale, scale_rel_tol: float):
    """Vendor output -> OpenCV camera frame.

    Returns X_cv [V,3] f64, R_cam_local [3,3] (column form: x_cv = R v_local * s + t_cv),
    t_cv [3], s (scalar). R_cam_local is returned for QC only (SKILL §3: never for B_tar)."""
    V = np.asarray(V_local, np.float64)
    assert V.ndim == 2 and V.shape[1] == 3, f"V_local must be [V,3], got {V.shape}"
    assert np.isfinite(V).all(), "non-finite vertices"
    s = read_scale(scale, scale_rel_tol)
    R = quat_wxyz_to_matrix(q_wxyz)
    t = np.asarray(t, np.float64).reshape(3)
    assert np.isfinite(t).all(), f"non-finite translation {t}"
    X = ((V * s) @ R + t) @ P3D_TO_CV
    R_cam_local = P3D_TO_CV @ R.T
    t_cv = t @ P3D_TO_CV
    assert np.isfinite(X).all()
    return X, R_cam_local, t_cv, s
