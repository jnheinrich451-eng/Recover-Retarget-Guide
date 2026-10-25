"""m2_a — Module 2 admission reader (skills/SKILL_m2_a_v2_thin_reader.md).

B.5 measures. m2_a gates. m2_a estimates nothing.

Three jobs: (1) re-gauge B.5's camera-0 quantities into the camera-t_a frame,
(2) evaluate the admission gates — every one, no short-circuit; a null lock
threshold means the statistic is recorded with pass=None (ungated, CLAUDE.md
§4), (3) emit the frozen stage0.npz + qc_m2a.json. Both files are written for
refused clips; only an upstream hash mismatch is an error.

Pure NumPy, f64 arithmetic, f32 on disk. No model. Reads contracts only:
docs/contract_stage_b5_v1_2.md, SKILL_stage_c_mask §7, contract_stage_a_v1.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

B5_SCHEMA = "b5_v1.2"
STAGE0_SCHEMA = "m2a_stage0_v1"
LOCK_KEYS = ("m2a_theta_max_deg", "m2a_kappa_min", "m2a_planarity_min",
             "m2a_spread_min", "m2a_mask_area_min_px", "m2a_truncation_max")
GATE_NAMES = ("g_b5_present", "g_present", "g_mask_area", "g_mask_trunc",
              "g_heading", "g_up", "g_basis")
OUTCOME_VOCAB = ("pass", "fail", "fallback", "ungated", "n/a")


class HashMismatch(RuntimeError):
    """Upstream artifact chain broken — the only hard error in this stage."""


# ----------------------------------------------------------------------------
# small geometry helpers
# ----------------------------------------------------------------------------
def _unit(v):
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.zeros(3)


def angle_rad(a, b):
    a, b = _unit(a), _unit(b)
    if not (a.any() and b.any()):
        return float("nan")
    return float(np.arccos(np.clip(a @ b, -1.0, 1.0)))


def kappa_basis(u, d):
    """‖u × d‖ = sin(angle(u, d)) = ‖u_perp‖ in m2_c's eq. (12)."""
    u, d = _unit(u), _unit(d)
    if not (u.any() and d.any()):
        return float("nan")
    return float(np.linalg.norm(np.cross(u, d)))


# ----------------------------------------------------------------------------
# config hash for this stage (its own lock keys + its yaml section)
# ----------------------------------------------------------------------------
def m2a_config_hash(thresholds: dict, m2a_section: dict) -> str:
    blob = json.dumps({"lock": {k: thresholds.get(k) for k in LOCK_KEYS},
                       "m2a": m2a_section}, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# ----------------------------------------------------------------------------
# §5 re-gauge — the only arithmetic in this stage
# ----------------------------------------------------------------------------
def regauge(G, t_a, d_src=None, u_scene=None, n_plane=None, d_plane=None):
    """Camera-0 -> camera-t_a. G[t] = [R | p] maps camera-t into camera-0, so
    directions go by R.T and the plane offset picks up n^(0)·p (frame-0
    normal, NOT the re-gauged one — SKILL §5). Returns dict of f64 arrays;
    None in -> zeros out (the zero-vector convention, §7)."""
    G = np.asarray(G, np.float64)
    R_a, p_a = G[t_a, :3, :3], G[t_a, :3, 3]
    out = {"R_a": R_a, "p_a": p_a}

    def rot(v):
        return R_a.T @ np.asarray(v, np.float64)

    out["d_src"] = rot(d_src) if d_src is not None else np.zeros(3)
    out["u_scene"] = rot(u_scene) if u_scene is not None else np.zeros(3)
    if n_plane is not None:
        n0 = np.asarray(n_plane, np.float64)
        out["n_plane"] = rot(n0)
        out["d_plane"] = float(d_plane) + float(n0 @ p_a)
    else:
        out["n_plane"], out["d_plane"] = np.zeros(3), 0.0
    # asserts: rotation preserves norms and the angle between heading and up
    for k in ("d_src", "u_scene", "n_plane"):
        if out[k].any():
            assert abs(np.linalg.norm(out[k]) - 1.0) < 1e-5, f"{k} not unit after re-gauge"
    if out["d_src"].any() and out["u_scene"].any():
        k0 = kappa_basis(u_scene, d_src)
        k1 = kappa_basis(out["u_scene"], out["d_src"])
        assert abs(k0 - k1) < 1e-6, f"kappa_basis not gauge-invariant: {k0} vs {k1}"
    return out


# ----------------------------------------------------------------------------
# §6 mask statistics
# ----------------------------------------------------------------------------
def mask_stats(M_ta):
    """Area (px, frac) and truncation. Truncation definition (recorded in qc,
    SKILL §12 open item resolved here explicitly): fraction of the mask's
    OUTLINE pixels (mask minus its 4-neighbour erosion) that lie on the
    1-px image border — i.e. how much of the silhouette is cut by the frame."""
    M = np.asarray(M_ta, bool)
    H, W = M.shape
    area = int(M.sum())
    if area == 0:
        return {"mask_area_px": 0, "mask_area_frac": 0.0, "mask_border_frac": float("nan"),
                "mask_touches_border": False}
    er = M.copy()
    er[1:] &= M[:-1]; er[:-1] &= M[1:]; er[:, 1:] &= M[:, :-1]; er[:, :-1] &= M[:, 1:]
    er[0, :] = er[-1, :] = er[:, 0] = er[:, -1] = False   # border pixels are outline
    outline = M & ~er
    ring = np.zeros_like(M)
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    n_out = int(outline.sum())
    n_cut = int((outline & ring).sum())
    return {"mask_area_px": area, "mask_area_frac": float(area / (H * W)),
            "mask_border_frac": float(n_cut / max(n_out, 1)),
            "mask_touches_border": bool(n_cut > 0)}


# ----------------------------------------------------------------------------
# §6 gates — evaluate all; null threshold => ungated (pass=None)
# ----------------------------------------------------------------------------
def _gate(stat: dict, threshold, cond):
    """cond(threshold) -> bool. None threshold -> ungated."""
    if threshold is None:
        return {**stat, "threshold": None, "pass": None, "outcome": "ungated"}
    ok = bool(cond(float(threshold)))
    return {**stat, "threshold": threshold, "pass": ok, "outcome": "pass" if ok else "fail"}


def evaluate_gates(th: dict, b5_ok: bool, b5_qc: dict | None, present_ta: bool,
                   mstats: dict, kappa: float):
    """Returns (gates dict, heading_defined, up_defined, admitted)."""
    g = {}
    g["g_b5_present"] = {"pass": bool(b5_ok), "outcome": "pass" if b5_ok else "fail"}
    g["g_present"] = {"pass": bool(present_ta), "outcome": "pass" if present_ta else "fail"}
    g["g_mask_area"] = _gate({"mask_area_px": mstats["mask_area_px"]},
                             th.get("m2a_mask_area_min_px"),
                             lambda t: mstats["mask_area_px"] >= t)
    bf = mstats["mask_border_frac"]
    g["g_mask_trunc"] = _gate({"mask_border_frac": bf}, th.get("m2a_truncation_max"),
                              lambda t: (not np.isnan(bf)) and bf <= t)

    q = b5_qc or {}
    heading_defined = bool(b5_ok and q.get("heading_defined", False))
    up_defined = bool(b5_ok and q.get("up_defined", False))
    sig_deg = float(np.degrees(q["sigma_theta"])) if (b5_ok and q.get("sigma_theta") is not None
                                                     and np.isfinite(q["sigma_theta"])) else float("nan")
    if not heading_defined:
        g["g_heading"] = {"sigma_theta_deg": sig_deg, "threshold": th.get("m2a_theta_max_deg"),
                          "pass": None, "outcome": "n/a"}
    else:
        g["g_heading"] = _gate({"sigma_theta_deg": sig_deg}, th.get("m2a_theta_max_deg"),
                               lambda t: np.isfinite(sig_deg) and sig_deg <= t)
        if g["g_heading"]["pass"] is False:
            heading_defined = False

    up_src = q.get("up_source")
    pl, sp = q.get("planarity"), q.get("spread")
    if not up_defined:
        g["g_up"] = {"up_source": up_src, "planarity": pl, "spread": sp,
                     "pass": None, "outcome": "n/a"}
    elif up_src != "plane":
        # B.5's fallback ladder already engaged (camera prior): recorded, not refused
        g["g_up"] = {"up_source": up_src, "planarity": pl, "spread": sp,
                     "pass": None, "outcome": "fallback"}
    else:
        tp, ts = th.get("m2a_planarity_min"), th.get("m2a_spread_min")
        if tp is None or ts is None:
            g["g_up"] = {"up_source": up_src, "planarity": pl, "spread": sp,
                         "threshold": [tp, ts], "pass": None, "outcome": "ungated"}
        else:
            ok = bool(pl is not None and sp is not None and pl >= float(tp) and sp >= float(ts))
            g["g_up"] = {"up_source": up_src, "planarity": pl, "spread": sp,
                         "threshold": [tp, ts], "pass": ok, "outcome": "pass" if ok else "fail"}
            if not ok:
                up_defined = False
    if not up_defined:
        heading_defined = False          # B_des needs both (SKILL §6)

    if not (heading_defined and up_defined):
        g["g_basis"] = {"kappa_basis": kappa, "threshold": th.get("m2a_kappa_min"),
                        "pass": None, "outcome": "n/a"}
    else:
        g["g_basis"] = _gate({"kappa_basis": kappa}, th.get("m2a_kappa_min"),
                             lambda t: np.isfinite(kappa) and kappa >= t)

    refusing = ("g_b5_present", "g_present", "g_mask_area", "g_mask_trunc", "g_basis")
    admitted = all(g[k]["pass"] is not False for k in refusing)
    return g, heading_defined, up_defined, admitted


# ----------------------------------------------------------------------------
# loaders + hash chain
# ----------------------------------------------------------------------------
def load_b5(b5_dir: Path, clip: str):
    """Returns (arrays or None, qc or None). Missing artifact -> (None, None)."""
    npz, qc = Path(b5_dir) / f"{clip}.npz", Path(b5_dir) / f"{clip}_qc_b5.json"
    if not (npz.exists() and qc.exists()):
        return None, None
    with np.load(npz) as z:
        arr = {k: z[k] for k in z.files}
    return arr, json.loads(qc.read_text(encoding="utf-8"))


def check_chain(bundle_hash: str, qc_c: dict, qc_b5: dict | None):
    """Every upstream artifact must agree on the Stage A bundle it derives
    from, and B.5 must have been built on the Stage C we are reading."""
    if qc_c.get("upstream_bundle_hash") != bundle_hash:
        raise HashMismatch(f"stage_c built on bundle {qc_c.get('upstream_bundle_hash')}, "
                           f"current bundle is {bundle_hash}")
    if qc_b5 is None:
        return
    if qc_b5.get("up_source") == "plane" and not qc_b5.get("plane_d_signed_with_n", False):
        raise HashMismatch("stage_b5 record predates the plane_d sign fix (2026-09-14) — regenerate B.5 "
                           "(scripts/run_stage_b5.py) before m2_a")
    if qc_b5.get("schema") != B5_SCHEMA:
        raise HashMismatch(f"stage_b5 schema {qc_b5.get('schema')!r} != {B5_SCHEMA!r} "
                           "— re-run B.5 v1.2")
    h = qc_b5.get("hashes", {})
    if h.get("bundle") != bundle_hash:
        raise HashMismatch(f"stage_b5 built on bundle {h.get('bundle')}, current {bundle_hash}")
    if h.get("stage_c") != qc_c.get("config_hash"):
        raise HashMismatch(f"stage_b5 built on stage_c {h.get('stage_c')}, "
                           f"current stage_c {qc_c.get('config_hash')}")


# ----------------------------------------------------------------------------
# the stage
# ----------------------------------------------------------------------------
def run_clip(clip: str, G, K, M, present, t_c: int, bundle_hash: str, qc_c: dict,
             b5_arr, b5_qc, thresholds: dict, config_hash: str):
    """Pure function: inputs already loaded. Returns (arrays, qc)."""
    check_chain(bundle_hash, qc_c, b5_qc)
    t_a = int(t_c)
    F = len(present)
    assert 0 <= t_a < F, f"t_c {t_a} outside [0,{F})"
    b5_ok = b5_arr is not None and b5_qc is not None
    present_ta = bool(present[t_a])
    mstats = mask_stats(M[t_a])

    d0 = b5_arr["d_src"] if (b5_ok and "d_src" in b5_arr) else None
    u0 = b5_arr["u_scene"] if b5_ok else None
    n0 = b5_arr["plane_n"] if (b5_ok and "plane_n" in b5_arr) else None
    dp0 = float(b5_arr["plane_d"][0]) if n0 is not None else None
    rg = regauge(G, t_a, d0, u0, n0, dp0)
    kappa0 = kappa_basis(u0, d0) if (u0 is not None and d0 is not None) else float("nan")
    kappa_a = kappa_basis(rg["u_scene"], rg["d_src"])

    gates, heading_defined, up_defined, admitted = evaluate_gates(
        thresholds, b5_ok, b5_qc, present_ta, mstats, kappa_a)

    # zero-vector convention: unit or exactly zero, never a third state
    d_src = rg["d_src"] if heading_defined else np.zeros(3)
    u_scene = rg["u_scene"] if up_defined else np.zeros(3)
    n_plane = rg["n_plane"] if (up_defined and n0 is not None) else np.zeros(3)
    d_plane = rg["d_plane"] if (up_defined and n0 is not None) else 0.0

    arrays = {"t_a": np.int32(t_a), "K_a": np.asarray(K[t_a], np.float32),
              "d_src": d_src.astype(np.float32), "u_scene": u_scene.astype(np.float32),
              "n_plane": n_plane.astype(np.float32), "d_plane": np.float32(d_plane),
              "admitted": np.bool_(admitted)}
    q = b5_qc or {}
    ungated = sorted(k for k, g in gates.items() if g.get("outcome") == "ungated")
    # 2026-09-15 (first proxy renders, clip 035): the heading pointed 16 deg OUT of the fitted ground
    # plane (upward) and the placed proxy tilted with it; a grounded subject moves within its plane.
    # Recorded, not gated (a threshold is the lock file's): angle between d_src and the plane.
    hoop = None
    if heading_defined and np.linalg.norm(n_plane) > 0:
        hoop = float(np.degrees(np.arcsin(np.clip(float(n_plane.astype(np.float64) @ d_src.astype(np.float64)), -1, 1))))
    qc = {"schema": STAGE0_SCHEMA, "clip_id": clip, "t_a": t_a, "t_c": int(t_c),
          "admitted": bool(admitted), "frame": "camera_t_a", "heading_out_of_plane_deg": hoop,
          "heading_defined": bool(heading_defined), "up_defined": bool(up_defined),
          "sigma_theta_deg": gates["g_heading"].get("sigma_theta_deg"),
          "up_source": q.get("up_source"), "slope_deg": q.get("angle_ua_ub_deg"),
          "planarity": q.get("planarity"), "spread": q.get("spread"),
          "kappa_basis": kappa_a, "kappa_basis_camera0": kappa0,
          "kappa_basis_deg": float(np.degrees(np.arcsin(np.clip(kappa_a, 0, 1)))) if np.isfinite(kappa_a) else None,
          **mstats,
          "truncation_definition": "fraction of mask outline pixels on the 1-px image border",
          "gate_outcomes": {k: gates[k]["outcome"] for k in GATE_NAMES},
          "gates": gates, "outcome_vocab": list(OUTCOME_VOCAB),
          "ungated_pending_calibration": ungated,
          "all_gates_ungated_or_pass": all(g["pass"] is not False for g in gates.values()),
          "b5_channel": q.get("channel"), "b5_heading_estimator": q.get("heading_estimator"),
          "b5_trajectory_source": q.get("traj_source"), "b5_churn_deg": q.get("churn_deg"),
          "b5_sigma_spec_deg": q.get("sigma_spec_deg"), "b5_support_frac": q.get("support_frac"),
          "thresholds_used": {k: thresholds.get(k) for k in LOCK_KEYS},
          "config_hash": config_hash,
          "upstream_hashes": {"stage_a": bundle_hash, "stage_c": qc_c.get("config_hash"),
                              "stage_b5": q.get("hashes", {}).get("config") if b5_ok else None,
                              "stage_b5_config": q.get("hashes", {}).get("stage_b5_config") if b5_ok else None}}
    return arrays, qc


# ----------------------------------------------------------------------------
# frozen writer / reader (the reader is m2_c's entry point)
# ----------------------------------------------------------------------------
def write_stage0(out_root: Path, clip: str, arrays: dict, qc: dict) -> Path:
    out = Path(out_root) / clip
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "stage0.npz", **arrays)
    (out / "qc_m2a.json").write_text(json.dumps(qc, indent=2, default=_json_default),
                                     encoding="utf-8")
    return out


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return str(o)


def read_stage0(clip_dir: Path) -> dict:
    """Consumer entry (m2_c). Validates the frozen schema; refuses otherwise."""
    d = Path(clip_dir)
    qc = json.loads((d / "qc_m2a.json").read_text(encoding="utf-8"))
    assert qc.get("schema") == STAGE0_SCHEMA, f"unexpected stage0 schema {qc.get('schema')!r}"
    with np.load(d / "stage0.npz") as z:
        a = {k: z[k] for k in z.files}
    for k in ("t_a", "K_a", "d_src", "u_scene", "n_plane", "d_plane", "admitted"):
        assert k in a, f"stage0.npz missing {k}"
    assert a["K_a"].shape == (3, 3) and a["K_a"].dtype == np.float32
    for k in ("d_src", "u_scene", "n_plane"):
        v = a[k].astype(np.float64)
        assert v.shape == (3,) and a[k].dtype == np.float32
        n = np.linalg.norm(v)
        assert n == 0.0 or abs(n - 1.0) < 1e-5, f"{k} must be unit or exactly zero (got {n})"
        assert np.isfinite(v).all()
    assert (np.linalg.norm(a["d_src"]) > 0) == bool(qc["heading_defined"])
    assert (np.linalg.norm(a["u_scene"]) > 0) == bool(qc["up_defined"])
    assert qc.get("frame") == "camera_t_a", f"stage0 frame is {qc.get('frame')!r}; consumers read camera_t_a only"
    a["frame"] = qc["frame"]                  # the frame travels WITH the vectors (m2_d's prompt reads it from here)
    a["qc"] = qc
    return a
