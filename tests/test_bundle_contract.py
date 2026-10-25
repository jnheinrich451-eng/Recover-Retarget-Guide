"""Contract tests for bundle writer/reader (SKILL §6) — no model needed.

Run:  python tests/test_bundle_contract.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess.bundle import read_bundle, run_gates, write_bundle  # noqa: E402
from src.preprocess.config import config_hash  # noqa: E402
from src.preprocess.tracks import eq24_relative, eq24_residual_px, project  # noqa: E402


def make_cfg(tmp: str) -> dict:
    cfg = {"paths": {"cache_dir": tmp},
           "d4rt": {"repo": "x", "checkpoint": "y", "dtype": "float32"},
           "video": {"process_height": 8, "max_frames": 4, "depth_stride": 1},
           "thresholds": {"tau_v": 0.5, "tau_c": 0.5, "tau_g_px": 2.0,
                          "tau_g2b_px": 2.5, "anchor_stride": 8,
                          "a_cov_min_tracks": 200, "a_cov_min_frac": 0.9}}
    cfg["_meta"] = {"config_hash": config_hash(cfg)}
    return cfg


def make_arrays(N=5, F=4, H=8, W=8) -> dict:
    rng = np.random.default_rng(1)
    return {
        "K": np.tile(np.eye(3, dtype=np.float32) * 4, (F, 1, 1)),
        "G": np.tile(np.eye(4, dtype=np.float32), (F, 1, 1)),
        "D": rng.random((F, H, W)).astype(np.float16),
        "anchors": np.column_stack([rng.random((N, 2)) * 7,
                                    np.zeros(N)]).astype(np.float32),
        "X_local": rng.random((N, F, 3)).astype(np.float32),
        "X0": rng.random((N, F, 3)).astype(np.float32),
        "P": rng.random((N, F, 2)).astype(np.float32),
        "V": rng.random((N, F)).astype(np.float32),
        "C": rng.random((N, F)).astype(np.float32),
        "Q": rng.random((N, F)) > 0.3,
    }


PROV = {"geometry_source": "test", "model_commit": "abc", "checkpoint": "ck",
        "gpu": "none", "env": "test"}
QC = {"gates": {}, "all_pass": True}


def expect_raise(fn, part):
    try:
        fn()
    except (AssertionError, RuntimeError) as e:
        assert part in str(e), f"wrong error: {e}"
        return
    raise SystemExit(f"expected failure containing '{part}' did not occur")


def test_gates():
    """Synthetic perfect geometry: A-G2 must report ~0 statistics; pass=None
    when lock thresholds are null (ungated), True when thresholds are set."""
    N, T = 300, 5
    rng = np.random.default_rng(0)
    K = np.tile(np.array([[100, 0, 128], [0, 100, 128], [0, 0, 1]], np.float32),
                (T, 1, 1))
    G = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
    G[:, 0, 3] = np.arange(T) * 0.1
    X0 = rng.uniform(-1, 1, (N, T, 3)).astype(np.float32)
    X0[..., 2] += 3
    Xl = X0 - G[None, :, :3, 3]
    P = project(K, Xl)
    Q = np.ones((N, T), bool)
    tr = {"Q": Q, "V": np.ones((N, T), np.float32), "C": np.ones((N, T), np.float32),
          "P": P, "X_local": Xl, "X0": X0,
          "anchors": np.column_stack([P[:, 0], np.zeros(N)]).astype(np.float32),
          "eq24_px": eq24_residual_px(K, G, Xl, X0, Q),
          "eq24_rel": eq24_relative(G, Xl, X0, Q),
          "sim3_scale": np.ones(T, np.float32)}
    th = {"tau_v": .5, "tau_c": .5, "tau_g_px": 30.0, "tau_g2b_px": 2.5,
          "anchor_stride": 8, "a_cov_min_tracks": 200, "a_cov_min_frac": 0.9,
          "tau_g_rel": None, "a_g2_min_inlier_frac": None}
    qc = run_gates(tr, {"thresholds": th})
    g2 = qc["gates"]["A-G2"]
    assert g2["pass"] is None and qc["ungated_pending_calibration"] == ["A-G2"]
    assert g2["median_rel"] < 1e-6
    assert qc["all_pass"], "gated checks must still pass on perfect data"
    qc2 = run_gates(tr, {"thresholds": {**th, "tau_g_rel": 0.05,
                                        "a_g2_min_inlier_frac": 0.5}})
    assert qc2["gates"]["A-G2"]["pass"] is True
    assert qc2["ungated_pending_calibration"] == []
    print("PASS A-G2 relative statistic (ungated + gated paths)")


def main():
    test_gates()
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_cfg(tmp)

        # round trip
        arrays = make_arrays()
        write_bundle(cfg, "clip_x", arrays, PROV, QC)
        back = read_bundle(cfg, "clip_x")
        for k in arrays:
            assert np.array_equal(back[k], arrays[k]), f"round-trip mismatch: {k}"
        print("PASS round-trip")

        # non-finite rejected (A-FIN)
        bad = make_arrays()
        bad["X0"][0, 0, 0] = np.nan
        expect_raise(lambda: write_bundle(cfg, "clip_nan", bad, PROV, QC), "non-finite")
        print("PASS A-FIN reject")

        # wrong dtype rejected
        bad = make_arrays()
        bad["D"] = bad["D"].astype(np.float32)
        expect_raise(lambda: write_bundle(cfg, "clip_dt", bad, PROV, QC), "dtype")
        print("PASS dtype reject")

        # inconsistent N rejected
        bad = make_arrays()
        bad["V"] = bad["V"][:3]
        expect_raise(lambda: write_bundle(cfg, "clip_n", bad, PROV, QC), "shape")
        print("PASS shape reject")

        # missing provenance key rejected (A-G8)
        expect_raise(lambda: write_bundle(cfg, "clip_p", make_arrays(),
                                          {"gpu": "x"}, QC), "provenance")
        print("PASS A-G8 reject")

        # stale config hash rejected on read
        cfg2 = make_cfg(tmp)
        cfg2["thresholds"]["tau_g_px"] = 99.0
        cfg2["_meta"]["config_hash"] = config_hash(cfg2)
        expect_raise(lambda: read_bundle(cfg2, "clip_x"), "stale")
        print("PASS stale-config reject")

    print("all bundle contract tests passed")


if __name__ == "__main__":
    main()
