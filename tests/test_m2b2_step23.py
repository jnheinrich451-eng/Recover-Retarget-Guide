"""m2_b2 build steps 2-3 — SKILL_m2_b2 v1.1 §10 tests 3, 4, 5, 6.

Run:  python tests/test_m2b2_step23.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.anchor.m2b2 import basis, refconst, silhouette                 # noqa: E402
from src.anchor.m2c import placement, proxy                             # noqa: E402
from src.anchor.m2c.synthetic import rot_x, rot_y                       # noqa: E402

REPORT = ["# m2_b2 steps 2-3 — tests 3, 4, 5, 6", ""]
H = W = 1024
K = np.array([[1024.0, 0, 512], [0, 1024.0, 512], [0, 0, 1]])
rng = np.random.default_rng(3)

# ---------------------------------------------------------------- test 3: facing sign
B_r, B_l = basis.b_tar_from_facing("right"), basis.b_tar_from_facing("left")
assert np.allclose(B_r, proxy.CANON_B_TAR), "right-facing B_tar must equal C1's canonical basis"
assert np.allclose(B_r[:, 0], -B_l[:, 0]) and np.allclose(B_r[:, 1], B_l[:, 1]) and np.allclose(B_r[:, 2], -B_l[:, 2])
assert abs(np.linalg.det(B_r) - 1) < 1e-12 and abs(np.linalg.det(B_l) - 1) < 1e-12
for bad in ("toward", "other"):
    try:
        basis.b_tar_from_facing(bad); raise AssertionError(bad)
    except basis.FacingRejected:
        pass
# the posed silhouette mirrors: same mesh, same heading, B_tar right vs left -> R_A differ by a
# 180-deg turn about up, and the silhouette is the horizontal mirror (perspective: nearly)
m = proxy.asym()                                          # lopsided, so a mirror is visible
d_src, u_scene = np.array([0.94, 0.0, 0.34]), np.array([0.0, -1.0, 0.0])
# weak-perspective camera: the 180-deg turn also flips depth, so under strong perspective the
# silhouette is NOT an exact mirror (measured at z = 8, f = 1024: mirrored IoU 0.879 < direct
# 0.908). At z = 64 with f = 8192 the same image size has +-2 % depth foreshortening.
K_far = np.array([[8192.0, 0, 512], [0, 8192.0, 512], [0, 0, 1]])
sils = {}
for name, B in (("right", B_r), ("left", B_l)):
    R_A, _ = placement.alignment(d_src, u_scene, B)
    X = np.array([0, 0, 64.0]) + proxy.aligned_vertices(m, R_A)
    sils[name] = silhouette.silhouette(X, m.F, K_far, H, W)
iou_same = silhouette.iou(sils["right"], sils["left"])
iou_mirror = silhouette.iou(sils["right"], sils["left"][:, ::-1])
assert iou_mirror > 0.95 and iou_same < iou_mirror - 0.02, (iou_same, iou_mirror)
R_r, _ = placement.alignment(d_src, u_scene, B_r); R_l, _ = placement.alignment(d_src, u_scene, B_l)
turn = np.degrees(np.arccos(np.clip((np.trace(R_r.T @ R_l) - 1) / 2, -1, 1)))
REPORT.append(f"test 3 facing sign: forward column flips, det +1 both, right == CANON_B_TAR; left vs right R_A differ by "
              f"{turn:.1f} deg about up; silhouettes: direct IoU {iou_same:.3f}, mirrored {iou_mirror:.3f}")

# ---------------------------------------------------------------- test 5: h_b percentile
V = proxy.ellipsoid().V
Vs = V.copy(); Vs[0] += np.array([0, 3.0, 0])             # one stray vertex 3 units BELOW (+y is down)
up = np.array([0.0, -1.0, 0.0])
hb_q, hb_min = refconst.h_b_percentile(V - V.mean(0), up, 0.01)
hb_q_s, hb_min_s = refconst.h_b_percentile(Vs - Vs.mean(0), up, 0.01)
assert hb_min_s > 2.9 and abs(hb_q_s - hb_q) < 5e-3, (hb_min_s, hb_q_s, hb_q)
rc = refconst.reference_constants(*refconst.normalise(Vs)[:1], proxy.ellipsoid().F, up, 0.01, 400)
assert "ref_hb_p01_minus_min" in rc and rc["ref_hb_p01_minus_min"] < -0.5
REPORT.append(f"test 5 h_b percentile: clean min {hb_min:.4f} p01 {hb_q:.4f} (p01 under-reads {100 * (1 - hb_q / hb_min):.1f} %); "
              f"one stray vertex 3 below: min {hb_min_s:.3f} (floats the object), p01 {hb_q_s:.4f} (unchanged); "
              f"QC field ref_hb_p01_minus_min recorded ({rc['ref_hb_p01_minus_min']:.3f} on the stray mesh)")

# ---------------------------------------------------------------- test 4: medoid determinism
consts = []
for k in range(5):
    jitter = 1 + 0.03 * rng.normal(size=3)
    Vk = proxy.ellipsoid().V * jitter * (1.4 if k == 2 else 1.0)          # seed 2 is the outlier
    Vn, _, _ = refconst.normalise(Vk)
    consts.append(refconst.reference_constants(Vn, proxy.ellipsoid().F, up, 0.01, 200))
feat = refconst.medoid_features(consts)
idx, tot = refconst.medoid(feat)
perm = rng.permutation(5)
idx_p, _ = refconst.medoid(feat[perm])
assert perm[idx_p] == idx, (idx, perm[idx_p])
assert idx != 2, "the scaled outlier seed must not be the medoid"
same = np.repeat(feat[:1], 4, 0)
assert refconst.medoid(same)[0] == 0                                   # identical seeds -> lowest index
assert refconst.medoid(feat[:1])[0] == 0
REPORT.append(f"test 4 medoid: seed {idx} chosen (outlier seed 2 not), invariant under permutation {perm.tolist()}; "
              f"identical seeds -> index 0; summed distances {np.round(tot, 2).tolist()}")

# ---------------------------------------------------------------- test 6: reference-constant isolation
keys = set(rc)
assert all(k.startswith("ref_") for k in keys), keys
assert not keys & {"a_0", "h_b", "delta_p", "a0", "hb", "dp"}
src = Path("src/anchor/m2c/placement.py").read_text(encoding="utf-8") + Path("src/anchor/m2c/proxy.py").read_text(encoding="utf-8")
assert "ref_" not in src, "m2_c must not read the ref_ namespace"
REPORT.append(f"test 6 isolation: record keys {sorted(keys)} all under ref_; m2_c sources contain no 'ref_'")

REPORT += ["", "all step-2/3 tests passed"]
Path("docs/report_m2b2_step23_tests.md").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
print("\n".join(REPORT))
