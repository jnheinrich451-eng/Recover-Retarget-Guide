"""m2_e build steps 4-5 — SKILL m2_e v2.2 §9 tests 4 (scan recovers a known yaw), 5 (mirror guard), 6 (keypoint facing sign,
on the body-facing probe's validated fixture), 8 (labelled pairs stay out). CPU, no models.

Run:  python tests/test_m2e_step4.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.anchor.m2c import compose, proxy                               # noqa: E402
from src.anchor.m2e import delta as D, qc as Q                          # noqa: E402

REPORT = ["# m2_e steps 4-5 — delta and QC: tests 4, 5, 6, 8", ""]
H = W = 256
K = np.array([[420.0, 0, 128.0], [0, 640.0, 128.0], [0, 0, 1.0]])        # anisotropic, as the controls
N = np.array([0.0, -1.0, 0.0]); X = np.array([0.1, 0.35, 4.0]); s = 0.6; R_A = np.eye(3)
mesh = proxy.make_proxy("asym")                                          # an asymmetric proxy: no 180-degree mirror pair

# ---- test 4: scan recovers a known yaw within the curve width, and theta = 0 reproduces the plain render's IoU
for theta_true in (12.0, -27.5, 0.0):
    _, anchor = compose.render_object(mesh.V, mesh.F, X, s, D.rot_about(N, theta_true) @ R_A, K, H, W)
    r = D.yaw_scan(mesh.V, mesh.F, X, s, R_A, N, K, anchor, scan_deg=45, step_deg=1.0, tol=0.01)
    assert abs(r["delta_deg"] - theta_true) <= max(0.5, r["curve_width_deg"] / 2), (theta_true, r["delta_deg"], r["curve_width_deg"])
    assert r["iou_peak"] > 0.97 and not r["peak_at_scan_edge"]
_, plain = compose.render_object(mesh.V, mesh.F, X, s, R_A, K, H, W)
r0 = D.yaw_scan(mesh.V, mesh.F, X, s, R_A, N, K, plain, scan_deg=45, step_deg=1.0)
assert abs(r0["iou_at_zero"] - 1.0) < 1e-9 and abs(r0["delta_deg"]) < 0.5
REPORT.append(f"test 4 scan: yaws 12, -27.5 and 0 deg recovered within the curve width (last: delta {r0['delta_deg']:+.2f}, width {r0['curve_width_deg']:.0f} deg, "
              f"peak IoU {r0['iou_peak']:.3f}); the theta = 0 point equals the plain render's IoU exactly (the same render)")

# ---- test 5: mirror guard — the anchor rendered at theta_true + 180: the restricted scan cannot return it
theta_true = 12.0
_, mirror = compose.render_object(mesh.V, mesh.F, X, s, D.rot_about(N, theta_true + 180.0) @ R_A, K, H, W)
rm = D.yaw_scan(mesh.V, mesh.F, X, s, R_A, N, K, mirror, scan_deg=45, step_deg=1.0)
assert -45.0 <= rm["delta_deg"] <= 45.0
r_true = D.yaw_scan(mesh.V, mesh.F, X, s, R_A, N, K, compose.render_object(mesh.V, mesh.F, X, s, D.rot_about(N, theta_true) @ R_A, K, H, W)[1], 45, 1.0)
ps_true = D.paired_scan(mesh.V, mesh.F, X, s, R_A, N, K, compose.render_object(mesh.V, mesh.F, X, s, D.rot_about(N, theta_true) @ R_A, K, H, W)[1], scan_deg=45, step_deg=1.0)
ps_flip = D.paired_scan(mesh.V, mesh.F, X, s, R_A, N, K, mirror, scan_deg=45, step_deg=1.0)
assert ps_true["front_back_margin"] > 0.05 and ps_flip["front_back_margin"] < -0.05, (ps_true["front_back_margin"], ps_flip["front_back_margin"])
REPORT.append(f"test 5 mirror guard: an anchor at {theta_true + 180:.0f} deg gives delta {rm['delta_deg']:+.1f} deg inside the +-45 window with peak IoU {rm['iou_peak']:.3f} "
              f"(vs {r_true['iou_peak']:.3f} for the true {theta_true:.0f} deg) — a plausible, quiet wrong answer; the absolute IoU is NOT a tell on real anchors (obj_fidelity 0.63-0.97). "
              f"PAIRED SCAN (reviewer 2026-09-23): front/back margin {ps_true['front_back_margin']:+.3f} for the true anchor, {ps_flip['front_back_margin']:+.3f} for the flipped one — the sign of the margin "
              "reads the flip; on real pairs g_front_back (m2_d's blind heading label vs the control class) is the complete guard")

# ---- test 6: keypoint facing sign on the probe's validated fixture (370.1.008: walks away, expected az +90 within 25 deg)
fx = json.loads(Path("measurements/body_facing/000000000370.1.008.json").read_text(encoding="utf-8"))
f, info = D.body_facing(np.asarray(fx["kp3d_cam"], float))
az = float(np.degrees(np.arctan2(f[2], f[0])))
assert abs(az - fx["az_l_cross_u"]) < 1e-6, (az, fx["az_l_cross_u"])                      # reproduces the probe's l x u to the digit
assert abs((az - fx["expected_az"] + 180) % 360 - 180) <= fx["tol"]                         # and agrees with the human's verdict
assert abs((az - fx["az_u_cross_l"] + 180) % 360 - 180) > 90                                # the other order points the opposite way
assert abs(info["shoulder_hip_angle_deg"] - fx["shoulder_hip_angle"]) < 1e-6
# body_delta: forward = R_A B_tar[:, 0]; with B_tar canonical (+x forward) and R_A rotating +x to the measured facing, delta must be ~0
B_tar = proxy.CANON_B_TAR
R_to_f = D.rot_about(N, D.signed_yaw_deg(B_tar[:, 0], f, N))
bd = D.body_delta(np.asarray(fx["kp3d_cam"], float), R_to_f, B_tar, N)
assert abs(bd["delta_deg"]) < 1e-6, bd["delta_deg"]
bd2 = D.body_delta(np.asarray(fx["kp3d_cam"], float), D.rot_about(N, -20.0) @ R_to_f, B_tar, N)
assert abs(bd2["delta_deg"] - 20.0) < 1e-6, bd2["delta_deg"]
REPORT.append(f"test 6 keypoints: l x u on the validated fixture gives azimuth {az:+.1f} deg (probe {fx['az_l_cross_u']:+.1f}, human's expected {fx['expected_az']:+.0f} +- {fx['tol']:.0f}); "
              f"u x l would give {fx['az_u_cross_l']:+.1f}; body_delta is 0 when B_des forward equals the facing and +20 when the proxy is turned 20 deg the other way")

# ---- test 8: labelled pairs stay out of accepted-set statistics
recs = [{"clip": "a", "category": "vehicle", "disposition": "accepted", "alpha": 1.0, "delta_deg_abs": 2.0, "r2_iou_vs_anchor_mask": 0.9},
        {"clip": "b", "category": "vehicle", "disposition": "accepted", "alpha": 3.0, "delta_deg_abs": 4.0, "r2_iou_vs_anchor_mask": 0.8},
        {"clip": "404", "category": "vehicle", "disposition": "labelled", "alpha": 100.0, "delta_deg_abs": 30.0, "r2_iou_vs_anchor_mask": 0.1}]
sm = Q.accepted_summary(recs)
assert sm["alpha"]["n"] == 2 and sm["alpha"]["max"] == 3.0 and sm["delta_deg_abs"]["max"] == 4.0 and sm["labelled_pairs_excluded"] == ["404/vehicle"]
try:
    Q.accepted_summary([{"clip": "x", "category": "y"}]); raise RuntimeError("a record without disposition was accepted")
except AssertionError:
    pass
REPORT.append("test 8: the accepted-set summary covers 2 of 3 records; the labelled pair is named as excluded and none of its values enters; a record without a disposition is refused")

REPORT += ["", "all m2_e step 4-5 tests passed"]
Path("docs/report_m2e_step45_tests.md").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
print("\n".join(REPORT))
