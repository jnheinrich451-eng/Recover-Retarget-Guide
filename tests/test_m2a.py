"""m2_a self-tests (SKILL_m2_a_v2_thin_reader §9). Synthetic; no clips.

Run:  python tests/test_m2a.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.anchor import m2a  # noqa: E402

F, H = 12, 32
NULL_TH = {k: None for k in m2a.LOCK_KEYS}


def rand_rot(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def make_G(rng, t_a):
    G = np.tile(np.eye(4), (F, 1, 1))
    G[t_a, :3, :3] = rand_rot(rng)
    G[t_a, :3, 3] = rng.normal(0, 3.0, 3)     # metres of camera travel
    return G


def synthetic_inputs(rng, t_a=5, heading=True, plane=True, present=True, area_frac=0.1,
                     touch_border=False):
    G = make_G(rng, t_a)
    K = np.tile(np.array([[40.0, 0, 16], [0, 40.0, 16], [0, 0, 1]]), (F, 1, 1))
    M = np.zeros((F, H, H), bool)
    s = int(np.sqrt(area_frac * H * H))
    r0 = 0 if touch_border else 8
    M[:, r0:r0 + s, 8:8 + s] = True
    pres = np.ones(F, bool)
    if not present:
        pres[t_a] = False; M[t_a] = False
    d0 = m2a._unit(rng.normal(size=3))
    u0 = m2a._unit(rng.normal(size=3))
    arr = {"u_scene": u0, "X_bar_track": np.zeros((F, 3))}
    qc = {"schema": m2a.B5_SCHEMA, "channel": "x0", "heading_defined": heading,
          "sigma_theta": 0.1, "up_defined": True, "up_source": "plane" if plane else "camera_prior", "plane_d_signed_with_n": True,
          "planarity": 0.95 if plane else None, "spread": 0.4 if plane else None,
          "angle_ua_ub_deg": 3.0, "hashes": {"bundle": "BH", "stage_c": "CH", "config": "AH",
                                             "stage_b5_config": "B5H"}}
    if heading:
        arr["d_src"] = d0
    if plane:
        arr["plane_n"] = u0
        arr["plane_d"] = np.array([-u0 @ rng.normal(0, 2, 3)])
    qc_c = {"config_hash": "CH", "upstream_bundle_hash": "BH"}
    return dict(G=G, K=K, M=M, present=pres, t_c=t_a, bundle_hash="BH", qc_c=qc_c,
                b5_arr=arr, b5_qc=qc)


def test_1_regauge_round_trip():
    rng = np.random.default_rng(1)
    G = make_G(rng, 4)
    d = m2a._unit(rng.normal(size=3))
    rg = m2a.regauge(G, 4, d_src=d, u_scene=m2a._unit(rng.normal(size=3)))
    back = G[4, :3, :3] @ rg["d_src"]
    assert np.linalg.norm(back - d) < 1e-6
    print("PASS 1 re-gauge round trip")


def test_2_plane_regauge_points():
    rng = np.random.default_rng(2)
    G = make_G(rng, 7)
    n0 = m2a._unit(rng.normal(size=3)); d0 = float(rng.normal())
    # 100 points on the plane in frame 0
    P0 = rng.normal(0, 5, (100, 3)); P0 -= ((P0 @ n0) + d0)[:, None] * n0[None, :]
    assert np.abs(P0 @ n0 + d0).max() < 1e-9
    rg = m2a.regauge(G, 7, n_plane=n0, d_plane=d0)
    R, p = G[7, :3, :3], G[7, :3, 3]
    Pa = (P0 - p) @ R                     # X^(a) = R.T (X^(0) - p)
    resid = np.abs(Pa @ rg["n_plane"] + rg["d_plane"])
    assert resid.max() < 1e-5, resid.max()
    # the WRONG formula (re-gauged normal) must fail, or the test proves nothing
    wrong = d0 + rg["n_plane"] @ p
    assert np.abs(Pa @ rg["n_plane"] + wrong).max() > 1e-3
    print(f"PASS 2 plane re-gauge (max |n·X+d| {resid.max():.1e}; wrong formula off by "
          f"{np.abs(Pa @ rg['n_plane'] + wrong).max():.2f})")


def test_3_angle_invariance():
    rng = np.random.default_rng(3)
    inp = synthetic_inputs(rng)
    arrays, qc = m2a.run_clip("c", **inp, thresholds=NULL_TH, config_hash="X")
    assert abs(qc["kappa_basis"] - qc["kappa_basis_camera0"]) < 1e-6
    print(f"PASS 3 kappa invariant ({qc['kappa_basis']:.4f})")


def test_4_zero_vector_convention():
    rng = np.random.default_rng(4)
    inp = synthetic_inputs(rng, heading=False)
    arrays, qc = m2a.run_clip("c", **inp, thresholds=NULL_TH, config_hash="X")
    assert not qc["heading_defined"]
    assert arrays["d_src"].dtype == np.float32 and np.all(arrays["d_src"] == 0)
    assert float(np.linalg.norm(arrays["d_src"])) == 0.0
    assert qc["gate_outcomes"]["g_basis"] == "n/a"
    print("PASS 4 zero-vector convention")


def test_5_gate_independence_and_ungated():
    rng = np.random.default_rng(5)
    th = dict(NULL_TH, m2a_mask_area_min_px=10_000)      # fails: mask is ~100 px
    inp = synthetic_inputs(rng)
    arrays, qc = m2a.run_clip("c", **inp, thresholds=th, config_hash="X")
    go = qc["gate_outcomes"]
    assert go["g_mask_area"] == "fail" and not qc["admitted"]
    assert all(v is not None for v in go.values()), go
    assert go["g_heading"] == "ungated" and go["g_basis"] == "ungated" and go["g_up"] == "ungated"
    assert "g_heading" in qc["ungated_pending_calibration"]
    assert qc["heading_defined"] and np.linalg.norm(arrays["d_src"]) > 0   # ungated keeps it
    # with everything null nothing refuses
    arrays2, qc2 = m2a.run_clip("c", **inp, thresholds=NULL_TH, config_hash="X")
    assert qc2["admitted"] and qc2["gate_outcomes"]["g_mask_area"] == "ungated"
    print(f"PASS 5 gate independence (all 7 populated on a failing clip; ungated admits): {go}")


def test_6_refusal_emits_and_camera_prior_fallback():
    rng = np.random.default_rng(6)
    inp = synthetic_inputs(rng, present=False, plane=False, touch_border=True)
    th = dict(NULL_TH, m2a_truncation_max=0.01)
    arrays, qc = m2a.run_clip("c", **inp, thresholds=th, config_hash="X")
    go = qc["gate_outcomes"]
    assert go["g_present"] == "fail" and not qc["admitted"]
    assert go["g_up"] == "fallback" and qc["up_defined"]          # camera prior: recorded, not refused
    assert np.all(arrays["n_plane"] == 0) and arrays["d_plane"] == 0
    with tempfile.TemporaryDirectory() as tmp:
        out = m2a.write_stage0(Path(tmp), "c", arrays, qc)
        back = m2a.read_stage0(out)
        assert bool(back["admitted"]) is False and back["qc"]["schema"] == m2a.STAGE0_SCHEMA
    print(f"PASS 6 refusal emits both files with valid schema: {go}")


def test_7_hash_mismatch_exits():
    rng = np.random.default_rng(7)
    for field, path in (("stage_c_vs_bundle", ("qc_c", "upstream_bundle_hash")),
                        ("b5_vs_bundle", ("b5_qc", "hashes", "bundle")),
                        ("b5_vs_stage_c", ("b5_qc", "hashes", "stage_c"))):
        inp = synthetic_inputs(rng)
        node = inp
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = "CORRUPT"
        try:
            m2a.run_clip("c", **inp, thresholds=NULL_TH, config_hash="X")
        except m2a.HashMismatch:
            continue
        raise AssertionError(f"{field}: hash mismatch did not raise")
    inp = synthetic_inputs(rng); inp["b5_qc"]["schema"] = "b5_v1.1"
    try:
        m2a.run_clip("c", **inp, thresholds=NULL_TH, config_hash="X")
        raise AssertionError("old B.5 schema accepted")
    except m2a.HashMismatch:
        pass
    print("PASS 7 hash chain: 3 mismatches + stale schema all raise")


def test_8_missing_b5_is_a_recorded_outcome():
    rng = np.random.default_rng(8)
    inp = synthetic_inputs(rng); inp["b5_arr"] = None; inp["b5_qc"] = None
    arrays, qc = m2a.run_clip("c", **inp, thresholds=NULL_TH, config_hash="X")
    assert qc["gate_outcomes"]["g_b5_present"] == "fail" and not qc["admitted"]
    assert not qc["heading_defined"] and not qc["up_defined"]
    assert np.all(arrays["d_src"] == 0) and np.all(arrays["u_scene"] == 0)
    print("PASS 8 missing B.5 -> g_b5_present fail, both files still emitted")


def test_9_truncation_statistic():
    M = np.zeros((H, H), bool); M[0:10, 10:20] = True          # touches top edge
    s = m2a.mask_stats(M)
    assert s["mask_touches_border"] and 0.2 < s["mask_border_frac"] < 0.35, s
    M2 = np.zeros((H, H), bool); M2[10:20, 10:20] = True
    s2 = m2a.mask_stats(M2)
    assert not s2["mask_touches_border"] and s2["mask_border_frac"] == 0.0
    print(f"PASS 9 truncation statistic (edge-cut square {s['mask_border_frac']:.2f}, interior 0)")


if __name__ == "__main__":
    test_1_regauge_round_trip()
    test_2_plane_regauge_points()
    test_3_angle_invariance()
    test_4_zero_vector_convention()
    test_5_gate_independence_and_ungated()
    test_6_refusal_emits_and_camera_prior_fallback()
    test_7_hash_mismatch_exits()
    test_8_missing_b5_is_a_recorded_outcome()
    test_9_truncation_statistic()
    print("\nall m2_a self-tests passed")
