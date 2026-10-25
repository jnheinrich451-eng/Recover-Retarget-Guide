"""B_tar from the facing verdict (SKILL §5 level 1, §6).

    B_tar = [f, u~, r],  r = f x u / |f x u|,  u~ = r x f / |r x f|

f = +x for "right", -x for "left" (the human's verdict on the proxy image); u = image up of the
reconstruction camera frame, (0, -1, 0) in OpenCV axes. Right-handed either way (det = +1);
with f = +x it equals `proxy.CANON_B_TAR`, with f = -x `placement.alignment` turns the proxy
180 deg about up — a rotation, not a mirror. "toward" / "other" are rejected candidates:
they carry an unknown angle, not a known sign. B_tar is ASSERTED from the prompt convention;
the pose head cannot supply it (E2 failed).
"""
from __future__ import annotations

import numpy as np

FACINGS = ("right", "left", "toward", "other")
ACCEPTED = ("right", "left")
U_IMAGE = np.array([0.0, -1.0, 0.0])          # image up in OpenCV camera axes


class FacingRejected(ValueError):
    """The candidate's facing gives no forward sign (toward / other) — reject the image."""


def b_tar_from_facing(facing: str, u_image=U_IMAGE) -> np.ndarray:
    assert facing in FACINGS, f"facing must be one of {FACINGS}, got {facing!r}"
    if facing not in ACCEPTED:
        raise FacingRejected(f"facing {facing!r} has no known forward sign — reject this candidate (SKILL §5)")
    f = np.array([1.0, 0, 0]) if facing == "right" else np.array([-1.0, 0, 0])
    u = np.asarray(u_image, np.float64)
    r = np.cross(f, u); r /= np.linalg.norm(r)
    ut = np.cross(r, f); ut /= np.linalg.norm(ut)
    B = np.stack([f, ut, r], axis=1)
    assert abs(np.linalg.det(B) - 1.0) < 1e-12, np.linalg.det(B)
    assert np.allclose(B.T @ B, np.eye(3), atol=1e-12)
    return B
