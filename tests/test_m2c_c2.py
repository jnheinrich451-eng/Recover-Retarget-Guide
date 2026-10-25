"""m2_c / C2 harness — steps 1–2 (SKILL_m2_c_C2 §8): generator verified against
the §1 closed form; ring fit + inversion (T1); cross-module plane agreement
against an m2_a-shaped stage0 (T11). Steps 3–6 follow human review.

Run:  python tests/test_m2c_c2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.anchor import m2a                                       # noqa: E402
from src.anchor.m2c import sanitise as S, bg_synthetic as B      # noqa: E402

PARAMS = {"r": 4, "w": 6, "N": 32, "sigma1": 2.0, "sigma2": 0.5, "fill_order": "fill_then_coarsen"}
REPORT = ["# m2_c C2 harness — steps 1–2 report", ""]


def log(s):
    REPORT.append(s); print(s)


def step1_generator():
    bg = B.make_background()
    H, W, K = bg["H"], bg["W"], bg["K"]
    # (a) the 3-D definition and the boxed affine formula agree to machine precision
    aff = S.eval_affine(bg["affine"], H, W)
    err = np.abs(aff - bg["zeta_true"]).max()
    assert err < 1e-14, err
    # (b) horizon inside the image, sky above it negative, ground below positive
    hz = np.where(np.diff(np.sign(bg["zeta_true"][:, W // 2])) != 0)[0]
    assert len(hz) == 1 and 0 < hz[0] < H // 2, hz
    assert (bg["zeta_true"][-1] > 0).all() and (bg["zeta_true"][0] < 0).all()
    # (c) depth round trip through f16 storage: zeta from stored depth vs true, inside valid ground
    zeta_back, ok = S.disparity(bg["depth16"], bg["valid"])
    ground = ok & ~bg["mask"]
    rel = np.abs(zeta_back[ground] - bg["zeta_true"][ground]) / bg["zeta_true"][ground]
    # (d) invalid = sky (zeta <= 0) plus the mask region carries the object, not the plane
    log(f"step 1: generator vs closed form max |affine - 3D| = {err:.1e}; horizon at row {hz[0]} of {H} "
        f"(sky above negative, ground below positive); f16 depth round trip max rel err on ground "
        f"{rel.max():.1e} (median {np.median(rel):.1e}); alpha,beta,gamma = {np.round(bg['affine'], 7).tolist()}; "
        f"sky (invalid) fraction {1 - ok.mean():.3f}")
    return bg, rel.max()


def T1_ring_fit(bg):
    H, W, K = bg["H"], bg["W"], bg["K"]
    zeta = bg["zeta_true"]
    valid = zeta > 0                                   # ground only (sky is invalid in reality)
    omega = S.dilate(bg["mask"], PARAMS["r"])
    ring = S.ring_of(omega, PARAMS["w"], valid & ~omega)
    coef, rec = S.ring_fit(zeta, ring, H, W)
    err = np.abs(coef - bg["affine"]).max() / np.abs(bg["affine"]).max()
    assert err < 1e-10, (coef, bg["affine"], err)
    n_hat, d_hat = S.plane_from_affine(coef, K)
    ang = np.degrees(np.arccos(np.clip(n_hat @ bg["n"], -1, 1)))
    assert ang < 1e-8 and abs(d_hat - bg["d"]) / bg["d"] < 1e-8, (ang, d_hat, bg["d"])
    # conditioning: compact ring O(1); a thin horizontal strip mask -> large
    strip = np.zeros((H, W), bool); strip[170:172, 60:200] = True
    ring_s = S.ring_of(S.dilate(strip, 1), 1, valid & ~S.dilate(strip, 1))
    _, rec_s = S.ring_fit(zeta, ring_s, H, W)
    assert rec_s["ring_cond"] > 20 * rec["ring_cond"], (rec["ring_cond"], rec_s["ring_cond"])
    log(f"T1 PASS ring fit: (alpha,beta,gamma) rel err {err:.1e} (tol 1e-10); inverted plane: normal "
        f"{ang:.1e} deg, d rel {abs(d_hat - bg['d']) / bg['d']:.1e} (tol 1e-8); ring {rec['ring_px']} px, "
        f"cond {rec['ring_cond']:.2f} (centred/scaled coords) vs thin strip {rec_s['ring_px']} px cond {rec_s['ring_cond']:.1f}")


def T11_plane_agreement(bg):
    """An m2_a-shaped stage0 (f32 by contract) carries n_plane, d_plane; C2's ring fit
    on the exact plane must agree with it."""
    H, W, K = bg["H"], bg["W"], bg["K"]
    st0 = {"t_a": np.int32(0), "K_a": K.astype(np.float32), "d_src": np.array([1, 0, 0], np.float32),
           "u_scene": (-bg["n"]).astype(np.float32) * -1, "n_plane": bg["n"].astype(np.float32),
           "d_plane": np.float32(bg["d"]), "admitted": np.bool_(True)}
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out = m2a.write_stage0(Path(tmp), "c2syn", st0, {"schema": m2a.STAGE0_SCHEMA, "heading_defined": True,
                                                          "up_defined": True, "clip_id": "c2syn"})
        back = m2a.read_stage0(out)
    zeta = bg["zeta_true"]; valid = zeta > 0
    omega = S.dilate(bg["mask"], PARAMS["r"])
    coef, _ = S.ring_fit(zeta, S.ring_of(omega, PARAMS["w"], valid & ~omega), H, W)
    n_hat, _ = S.plane_from_affine(coef, back["K_a"].astype(np.float64))
    n32 = back["n_plane"].astype(np.float64)
    # atan2(|a x b|, a.b) — arccos(a.b) is unusable near 0 deg: an f32-rounded unit vector has
    # |n| = 1 +- 3e-8 and arccos turns that into a spurious 0.01 deg (measured 2026-09-11)
    ang = float(np.degrees(np.arctan2(np.linalg.norm(np.cross(n_hat, n32)), n_hat @ n32)))
    assert ang < 1e-4, ang
    log(f"T11 PASS cross-module plane agreement: C2 ring-fit normal vs m2_a stage0 n_plane (f32 contract) = "
        f"{ang:.1e} deg (tol 1e-4)")


if __name__ == "__main__":
    bg, _ = step1_generator()
    T1_ring_fit(bg)
    T11_plane_agreement(bg)
    REPORT.append("")
    REPORT.append("Checkpoint (SKILL §8 step 1/2): report before continuing to the residual solve (T2, T3).")
    Path("docs/report_m2c_c2_steps12.md").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
    print("\nC2 steps 1-2 OK -> docs/report_m2c_c2_steps12.md")
