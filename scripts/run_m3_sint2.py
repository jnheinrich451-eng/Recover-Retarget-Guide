#!/usr/bin/env python
"""Module 3, S-int step 2 (brief v2.5 §2.2-§2.4, §3.3-§3.5, §9): camera path, object path, coverage over time, anchor schedule.

CPU only, Phase 1 (no Wan, no depth model). Reads cache/m3/<clip>/<category>/{lift.npz, contact_decision.json} from step 1 and
the contact check, the ViPE poses and sigma_hat record (A.5), Stage A's G and B.5's centroid track. Writes
cache/m3/<clip>/<category>/geometry.npz and reports/m3_sint/sint_step2.{md,json}. Read-only on every Module 1 / 2 cache.

Rules applied (none decided here):
  E'_t            ViPE cam2world, rebased to t_a, translations / sigma_hat(t_a)                      (D-M3-07 (a); one-frame bridge)
  object path     centroid in camera-t coordinates via D4RT's own G (bookkeeping), moved to the world with the SAME bridged
                  ViPE pose; drift model: x0 -> Stage B's (as delivered in B.5's track), branch -> none; g_t never applied here
  contact         G_tar and its displacement scaled by s_c from contact_decision.json (1 unless the pre-registered rule re-gauged)
  rotation        R_t = I (lock rotation.source is null until the S1-rot probe reports); object mode therefore centroid only

    python scripts/run_m3_sint2.py
    python scripts/run_m3_sint2.py --no-regauge        # the un-re-gauged sub-row (s_c forced to 1), written beside the main record
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.generate.m3 import contact, handoff as H                                   # noqa: E402
from src.generate.m3.geometry import camera, project, rasterize, vipe_camera        # noqa: E402

ORTHO_TOL = 1e-4          # rigidity check on stored float32 poses (interface assert, not a gate)
EPS = 1e-6                # brief §5: epsilon of the normalised fields


def voxel_downsample(V: np.ndarray, frac: float) -> np.ndarray:
    """One vertex per voxel of side frac x the mesh's largest extent (brief §3.2, lock lift.obj_voxel_frac)."""
    side = frac * float((V.max(0) - V.min(0)).max())
    key = np.floor((V - V.min(0)) / side).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return V[np.sort(idx)]


def angle_deg(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def run_pair(root: Path, lp: Path, lock: dict, regauge: bool, bridge_run: str = "rec", camera_src: str = "vipe_a5") -> dict:
    clip, cat = lp.parent.parent.name, lp.parent.name
    pair = f"{clip}/{cat}"
    h = json.loads((root / "cache" / "m2_e" / clip / cat / "handoff.json").read_text(encoding="utf-8"))
    t_a = int(h["t_a"]); F_pair, T_l = camera.frames_for_pair(h["n_frames_after_t_a"])
    src_frames = t_a + 4 * np.arange(T_l)
    with np.load(lp) as z:
        X_bg, V_obj, Kp, sky = z["X_bg"].astype(np.float64), z["V_obj"].astype(np.float64), z["Kp"].astype(np.float64), z["sky_lattice"].astype(bool)
    lat = lock["lattice"]; Hp, Wp = int(lat["Hp"]), int(lat["Wp"])
    rec = {"pair": pair, "disposition": h["disposition"], "F_pair": F_pair, "T_l": T_l, "t_a": t_a}

    # ---- contact decision (pre-registered rule; John 2026-10-06)
    dp = lp.parent / "contact_decision.json"
    assert dp.exists(), f"{dp} missing: run scripts/check_m3_contact.py first"
    dec = json.loads(dp.read_text(encoding="utf-8"))
    s_c = float(dec["s_c"]) if regauge else 1.0
    rec["contact"] = {"outcome": dec["outcome"], "s_c_rule": dec["s_c"], "s_c_applied": s_c, "edge_offset_rows": dec.get("edge_offset_rows")}

    if camera_src == "camera_c_ba":
        # ---- the census camera of record (plan v1; docs/report_m3_camera.md): camera C + bundle adjustment, gauge fixed at t_a by the
        # D4RT lift prior -> already the source gauge: no sigma_hat, no bridge (beta = 1); the object's camera-relative position is in the
        # same gauge (s_delta = 1; depth-preserving pairs, contract §2.4); the contact re-gauge s_c applies as on every pair
        cf = root / "cache" / "m3" / clip / "camera_c_ba.npz"
        assert cf.exists(), f"{cf} missing (the census camera gate of record; scripts/refine_m3_camera_ba.py --census)"
        with np.load(cf) as z:
            E_ba = z["E_c"].astype(np.float64); t_a_c = int(z["t_a"]); F_c = int(z["F_pair"])
        assert t_a_c == t_a, f"camera_c_ba t_a {t_a_c} != handoff t_a {t_a}"
        assert F_c >= F_pair, f"camera_c_ba holds {F_c} frames < F_pair {F_pair}"
        E_full = E_ba[:F_pair].copy()
        assert np.isfinite(E_full).all(), "camera_c_ba has unsolved frames inside the pair's window"
        assert np.allclose(E_full[0], np.eye(4), atol=1e-9), "camera_c_ba E[0] is not the identity (world = camera t_a)"
        R_ = E_full[:, :3, :3]; err_ = np.abs(R_.transpose(0, 2, 1) @ R_ - np.eye(3)).max()
        assert err_ <= ORTHO_TOL, f"camera_c_ba rotations not rigid: {err_:.2e}"
        E_l = E_full[::4]
        beta, s_delta = 1.0, 1.0
        sig = None; crec = {}
        rec["bridge"] = {"run": bridge_run, "beta": 1.0, "camera": "camera_c_ba", "s_delta": 1.0, "contact_undefined": None,
                         "note": "camera C + BA is in the source gauge by the lift prior at t_a: no bridge; s_delta = 1 (depth-preserving); s_c applied below as on every pair"}
        br = None
    else:
        # ---- Phase-2 camera from ViPE (A.5), one-frame gauge bridge
        pd_ = root / "bakeoff" / "stage_a5" / clip / "vipe" / "pose"
        pf = sorted(pd_.glob("*.npz"))
        assert pf, f"no ViPE pose under {pd_} (g_vipe_present)"
        with np.load(pf[0]) as z:
            P_c2w, inds = z["data"].astype(np.float64), z["inds"]
        assert list(inds[: t_a + F_pair]) == list(range(t_a + F_pair)), "ViPE frame indices not contiguous over the pair's window (A.5 V3)"
        crec = json.loads((root / lock["trajectory"]["sigma_t_record"]).read_text(encoding="utf-8"))
        assert clip in crec, f"{clip} not in the calibration record"
        sig = crec[clip]["sigma_abs"]
        E_full, E_l = vipe_camera.phase2_camera(P_c2w, t_a, sig[t_a], F_pair, ORTHO_TOL)
        br = lock["trajectory"].get("bridge")
    # ---- v2.6 §2.3-§2.4 (D-M3-08 amended): the camera carries beta on top of sigma_hat, the path carries s_delta, the geometry carries s_c
    if camera_src == "camera_c_ba":
        pass
    elif br is not None:
        assert br == "contact_carried", br
        # the record is written by check_m3_bridge_interval.py to <drive-root>/reports/m3_sint/; the lock names the repo copy. On Colab the
        # Drive copy is the live one; the repo copy exists only after John copies it back. First existing wins, and the record says which.
        cands = [root / "reports" / "m3_sint" / Path(lock["trajectory"]["bridge_record"]).name, Path(lock["trajectory"]["bridge_record"])]
        brec = next((c for c in cands if c.exists()), None)
        assert brec is not None, f"bridge record missing at {cands} (run scripts/check_m3_bridge_interval.py under the lock's bridge_camera first)"
        cam_b = lock["trajectory"].get("bridge_camera", "vipe_over_sigma")
        row = next((x for x in json.loads(brec.read_text(encoding="utf-8")) if x.get("pair") == pair and x.get("camera") == cam_b and "error" not in x), None)
        assert row is not None, f"{pair}: no bridge row under camera {cam_b} in {brec}"
        beta = row["record"]["runs"].get(bridge_run)
        assert beta is not None, f"{pair}: beta_{bridge_run} is not determined under {cam_b} (v2.6 §2.3: the pair does not enter S4)"
        s_delta = float(row["s_delta"])
        E_full = E_full.copy(); E_l = E_l.copy(); E_full[:, :3, 3] /= beta; E_l[:, :3, 3] /= beta
        rec["bridge"] = {"run": bridge_run, "beta": float(beta), "camera": cam_b, "s_delta": s_delta, "contact_undefined": bool(row["contact_undefined"]),
                         "interval": row["record"]["interval"], "record_file": str(brec)}
    else:
        beta, s_delta = None, s_c                                            # pre-amendment behaviour (v2.5): the 10-06c rule
        rec["bridge"] = {"run": None, "beta": None, "s_delta": s_delta, "note": "lock has no trajectory.bridge: v2.5 path scale (s_c on the displacement)"}
    rec["camera"] = {"source": camera_src if camera_src == "camera_c_ba" else lock["trajectory"]["source"], "sigma_ta": (None if sig is None else sig[t_a]),
                     "record_degenerate": (None if sig is None else bool(crec[clip]["degenerate"])),
                     "cam_path_len": float(np.linalg.norm(np.diff(np.linalg.inv(E_full)[:, :3, 3], axis=0), axis=1).sum()),
                     "cam_rot_total_deg": angle_deg(E_full[-1, :3, :3] @ np.array([0, 0, 1.0]), np.array([0, 0, 1.0]))}

    # ---- object path: B.5's camera-0 centroid -> camera-t via D4RT's own G -> world via the bridged ViPE pose
    with np.load(root / "cache" / "stage_b5" / f"{clip}.npz") as z:
        Xbar = z["X_bar_track"].astype(np.float64)
    qb = json.loads((root / "cache" / "stage_b" / f"{clip}_qc_b.json").read_text(encoding="utf-8"))
    channel = "branch" if qb.get("follow_cam", {}).get("branch_active") else "x0"
    with np.load(root / "cache" / "stage_a" / clip / "bundle.npz") as z:
        G = z["G"].astype(np.float64)                                           # G_{0<-t}: camera-t -> camera-0 (Stage A contract)
    assert np.isfinite(Xbar[src_frames]).all(), f"B.5 centroid track is NaN on latent frames {src_frames[~np.isfinite(Xbar[src_frames]).all(1)].tolist()} (absent, never filled)"
    m3cfg = __import__("yaml").safe_load(Path("configs/m3.yaml").read_text(encoding="utf-8"))["m3"]
    full = t_a + np.arange(F_pair)
    assert np.isfinite(Xbar[full]).all(), "B.5 centroid track has NaN inside the pair's window (absent, never filled)"
    Xw_full_b5 = vipe_camera.object_path_target(Xbar[full], G[full], E_full, s_delta)   # B.5's stitched world track through D4RT's G: the cross-check column (2026-10-11)
    c5p = root / "cache" / "stage_c5" / f"{clip}.npz"
    assert c5p.exists(), f"{c5p} missing: the object path of record is built from the same-frame C.5 lift (reviewer 2026-10-11 decision 1)"
    with np.load(c5p) as z:
        Xl_c5, P_c5, Q_c5 = z["X_local_dense"].astype(np.float64), z["P_dense"].astype(np.float64), z["Q_dense"].astype(bool)
    with np.load(root / "cache" / "stage_a" / clip / "bundle.npz") as z:
        K_a_all, Xl_a, Q_a = z["K"].astype(np.float64), z["X_local"].astype(np.float64), z["Q"].astype(bool)
    with np.load(root / "cache" / "stage_b" / f"{clip}.npz") as z:
        y_dyn_a = z["y_dyn"].astype(bool)
    n_vis_c5 = Q_c5[:, full].sum(0)
    assert (n_vis_c5 >= int(m3cfg["object_path"].get("c5_min_tracks", 10))).all(), f"C.5 object tracks visible per frame fall to {int(n_vis_c5.min())} inside the window"
    Xcam_c5 = np.stack([np.median(Xl_c5[Q_c5[:, t], t], axis=0) for t in full])          # [F_pair, 3] the object's same-frame camera position (lift gauge)
    # the boundary check on the static background: r_t = median z_t / z_{t-1} over static tracks visible at both frames (the lift's own gauge step)
    static_a = ~y_dyn_a
    r_t = np.ones(F_pair)
    for k in range(1, F_pair):
        t0_, t1_ = full[k - 1], full[k]
        sel_ = static_a & Q_a[:, t0_] & Q_a[:, t1_] & (Xl_a[:, t0_, 2] > 0.05) & (Xl_a[:, t1_, 2] > 0.05)
        r_t[k] = float(np.median(Xl_a[sel_, t1_, 2] / Xl_a[sel_, t0_, 2])) if sel_.sum() >= 20 else 1.0
    cc_path = root / "reports" / "m3_sint" / "camera_check.json"
    bounds = []
    if cc_path.exists():
        rows_cc = [x for x in json.loads(cc_path.read_text(encoding="utf-8")) if x.get("pair", "").startswith(clip)]
        bounds = [int(b) for b in (rows_cc[-1].get("boundaries_in_window") or [])] if rows_cc else []
    logr = np.abs(np.log(r_t[1:])); med_step = float(np.median(logr)) if len(logr) else 0.0
    bstep = {int(b): float(np.log(r_t[b - t_a])) for b in bounds if 0 < b - t_a < F_pair}
    scale_corr = np.ones(F_pair)
    applied_b = []
    for b, lr in bstep.items():
        if abs(lr) > 3.0 * max(med_step, 1e-9):                                     # a measured step: undo it for every frame from the boundary on
            scale_corr[b - t_a:] *= np.exp(-lr); applied_b.append(b)
    Xcam_c5_corr = Xcam_c5 * scale_corr[:, None]
    Gid = np.broadcast_to(np.eye(4), (F_pair, 4, 4))
    Xw_full = vipe_camera.object_path_target(Xcam_c5_corr, Gid, E_full, s_delta)        # same-frame camera position: G = I (D4RT's world frame never enters)
    # registered (reviewer): (a) no latent step at a boundary larger than 3x the path's median latent step; (b) the un-retargeted path reprojected
    # through the source camera reproduces the C.5 2-D centroid within 0.5 patch at every frame
    lat_steps = np.linalg.norm(np.diff(Xw_full[::4], axis=0), axis=1)
    med_lat = float(np.median(lat_steps)) if len(lat_steps) else 0.0
    b_lat = {int(b): float(lat_steps[min(max((b - t_a) // 4, 0), len(lat_steps) - 1)]) for b in bounds if 0 <= b - t_a < F_pair}
    reg_a = all(v <= 3.0 * max(med_lat, 1e-9) for v in b_lat.values())
    patch256 = 256.0 / int(lock["lattice"]["Wp"])
    err_px = []
    for k, t in enumerate(full):
        vq = Q_c5[:, t] & (Xl_c5[:, t, 2] > 1e-6)
        pk = (K_a_all[t] @ Xl_c5[vq, t].T).T; uv = pk[:, :2] / pk[:, 2:3]
        err_px.append(float(np.median(np.linalg.norm(uv - P_c5[vq, t], axis=1))) if vq.sum() >= 10 else np.nan)
    err_patch = np.array(err_px) / patch256                                       # per-track reprojection residual, median over tracks, per frame
    bnd_frames = [f_ for f_ in range(44, 53) if 0 <= f_ - t_a < F_pair]
    in_b = np.zeros(F_pair, bool); in_b[[f_ - t_a for f_ in bnd_frames]] = True
    med_in = float(np.nanmedian(err_patch[in_b])) if in_b.any() else None; med_out = float(np.nanmedian(err_patch[~in_b]))
    mad_out = float(1.4826 * np.nanmedian(np.abs(err_patch[~in_b] - med_out)))
    reg_b = bool(med_in is None or med_in <= med_out + 3.0 * max(mad_out, 1e-6))  # the boundary frames do not exceed the rest
    b5_steps = np.linalg.norm(np.diff(Xw_full_b5[::4], axis=0), axis=1)
    rec["object_path_c5"] = {"source": "same-frame C.5 lift median, G = I (reviewer 2026-10-11 decision 1)", "n_tracks_visible_min": int(n_vis_c5.min()),
                             "boundaries_in_window": bounds, "static_pair_ratio_log_at_boundaries": bstep, "median_abs_log_step": med_step,
                             "per_window_scale_applied_at": applied_b, "scale_corr_last": float(scale_corr[-1]),
                             "registered_a_no_boundary_step_over_3x_median": {"held": bool(reg_a), "latent_step_at_boundaries": b_lat, "median_latent_step": med_lat},
                             "registered_b_boundary_frames_do_not_exceed_rest": {"held": reg_b, "statistic": "per-track reprojection residual (lift through K_t vs own 2-D track), median over tracks per frame",
                                                                                   "median_patch_frames_44_52": med_in, "median_patch_other_frames": med_out, "mad_other_frames": mad_out,
                                                                                   "per_frame_44_52": [float(v) for v in err_patch[in_b]], "max_err_patch": float(np.nanmax(err_patch))},
                             "b5_cross_check": {"max_dev_from_c5_world": float(np.linalg.norm(Xw_full - Xw_full_b5, axis=1).max()),
                                                "b5_max_latent_step_over_median": float(b5_steps.max() / max(np.median(b5_steps), 1e-9)) if len(b5_steps) else None,
                                                "c5_max_latent_step_over_median": float(lat_steps.max() / max(med_lat, 1e-9)) if len(lat_steps) else None}}
    print(f"[sint2] {clip}/{cat}: C.5 path; boundaries {bounds}; log static ratio at boundaries {{{', '.join(f'{b}: {v:+.3f}' for b, v in bstep.items())}}} (median |log step| {med_step:.4f}); "
          f"per-window scale applied at {applied_b}; registered a {'held' if reg_a else 'FAILED'}, b {'held' if reg_b else 'FAILED'} (per-track reprojection median: frames 44-52 {med_in if med_in is None else round(med_in, 3)} vs other {med_out:.3f} +- {mad_out:.3f} patch); "
          f"B.5 max step/median {rec['object_path_c5']['b5_cross_check']['b5_max_latent_step_over_median']}", flush=True)
    Xw_raw = Xw_full[::4]
    sm = dict(m3cfg["object_path"])
    inv_lock = lock.get("object_path") or {}                      # the two invariants are decision tolerances: the lock wins when John set them (2026-10-06)
    sm.update({k: inv_lock[k] for k in ("invariant_net_disp_patch", "invariant_heading_deg") if k in inv_lock})
    from scipy.signal import savgol_filter
    # one window, two consumers (reviewer 2026-10-06h): the rotation estimator's lock key wins when set, else the config value
    sg_src = lock.get("rotation", {}).get("sg_window")
    w = min(int(sg_src if sg_src is not None else sm["sg_window"]), F_pair if F_pair % 2 == 1 else F_pair - 1)
    if w % 2 == 0:
        w -= 1
    Xw_sm_full = savgol_filter(Xw_full, w, int(sm["polyorder"]), axis=0) if w > int(sm["polyorder"]) else Xw_full.copy()
    Xw_sm = Xw_sm_full[::4]
    # pre-registered invariants, AMENDED by the reviewer 2026-10-06i: the raw endpoint was the wrong reference (it carries the jitter).
    #   (i) residual bias at the ends: r_t = c_raw - c_smooth; |mean r| over the first window and over the last window each < tol (patches,
    #       in projection); (ii) heading over the B.5 window: tolerance max(2 deg, sigma_theta of this clip); (iii) report max|r| and lag-1
    #       autocorrelation of r (a strongly correlated residual means the smoother removed structure, not noise) - report, not gate.
    r_full = Xw_full - Xw_sm_full
    pw = lambda X: project.project(X[None], E_l[:1], Kp)[0][0]            # noqa: E731  (patch units in the frame-0 camera)
    r_px = pw(Xw_full) - pw(Xw_sm_full)
    wl = min(w, F_pair)
    bias_first = float(np.linalg.norm(r_px[:wl].mean(0))); bias_last = float(np.linalg.norm(r_px[-wl:].mean(0)))
    kw_ = min(12, F_pair - 1)
    d_src_ = np.asarray(h["d_src"], np.float64)
    W_inv = min(16, F_pair - 1)
    p_sm, p_raw = pw(Xw_sm_full), pw(Xw_full)
    d_s, d_r = p_sm[W_inv] - p_sm[0], p_raw[W_inv] - p_raw[0]                      # projected displacements over W (patches)
    th_sr = np.radians(angle_deg(d_s, d_r)) if np.linalg.norm(d_s) > 1e-9 and np.linalg.norm(d_r) > 1e-9 else 0.0
    e_perp = float(np.linalg.norm(d_s) * abs(np.sin(th_sr)))                      # the heading difference in POSITION units (reviewer 2026-10-11 round 4)
    inv_head = float(np.degrees(th_sr))                                           # reported in degrees too
    qb5 = json.loads((root / "cache" / "stage_b5" / f"{clip}_qc_b5.json").read_text(encoding="utf-8"))
    sig_th = float(np.degrees(qb5["sigma_theta"])) if qb5.get("sigma_theta") is not None and np.isfinite(qb5["sigma_theta"]) else None
    head_tol = max(float(sm["invariant_heading_deg"]), sig_th if sig_th is not None else 0.0)
    rn = np.linalg.norm(r_px, axis=1); ac1 = float(np.corrcoef(rn[:-1], rn[1:])[0, 1]) if len(rn) > 3 and rn.std() > 0 else None
    head_vs_dsrc = {"raw": angle_deg(Xw_full[kw_] - Xw_full[0], d_src_), "smoothed": angle_deg(Xw_sm_full[kw_] - Xw_sm_full[0], d_src_)}   # reviewer 2026-10-06k: what a refusal MEANS (reported, not gated)
    smooth_ok = bias_first < float(sm["invariant_net_disp_patch"]) and bias_last < float(sm["invariant_net_disp_patch"]) and e_perp <= float(sm["invariant_net_disp_patch"])
    inv_net = max(bias_first, bias_last)
    print(f"[sint2] {clip}/{cat}: smoothing {'APPLIED' if smooth_ok else 'refused'}: e_perp {e_perp:.3f} patch (tol {float(sm['invariant_net_disp_patch'])}), end bias {bias_first:.3f} / {bias_last:.3f}, heading vs d_src (reported) {head_vs_dsrc['smoothed']:.1f} deg", flush=True)
    Xw = Xw_sm if smooth_ok else Xw_raw
    dc = Xw - Xw[0]                                                          # v2.6 §2.4: s_delta is already inside Xw (object_path_target), never applied twice
    rough = lambda P: float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum() / max(np.linalg.norm(P[-1] - P[0]), 1e-12))   # noqa: E731
    # registered (reviewer 2026-10-11 decision 2): the smoothed gross path stays within 0.3 patch of the raw one (in projection)
    smooth_dev = float(np.linalg.norm(r_px, axis=1).max())
    rec["path_smoothing"] = {"method": sm["smooth"], "sg_window": w, "heading_vs_d_src_deg_12f": head_vs_dsrc, "invariant": "residual bias at the ends (reviewer 2026-10-06i)", "tolerance_source": "lock m3.object_path" if inv_lock else "configs/m3.yaml", "net_disp_tol_patch": float(sm["invariant_net_disp_patch"]), "bias_first_window_patch": bias_first, "bias_last_window_patch": bias_last, "heading_tol_deg": head_tol, "sigma_theta_deg": sig_th, "resid_max_patch": float(rn.max()), "resid_lag1_autocorr": ac1, "sg_window_source": "lock.rotation.sg_window" if sg_src is not None else "configs/m3.yaml", "delta_c0_is_zero": bool(np.all(dc[0] == 0)), "applied": bool(smooth_ok), "invariant_net_disp_patch": inv_net,
                             "invariant_heading_deg": inv_head, "invariant_heading_perp_patch": e_perp, "invariant_window_frames": int(W_inv),
                             "heading_reference": "smoothed vs raw of the same path over 16 frames, in position units (reviewer 2026-10-11 round 4); d_src agreement reported, not gated",
                             "path_over_net_raw": rough(Xw_raw), "path_over_net_smooth": rough(Xw_sm), "max_dev_smooth_vs_raw_patch": smooth_dev}
    # D4RT-only displacement (brief §2.2), for the record: the two cameras' disagreement shows up here
    Rta = G[t_a, :3, :3]
    dc_d4rt = (Xbar[src_frames] - Xbar[t_a]) @ Rta * s_delta                                   # R_ta^T (c_t - c_ta)
    path = float(np.linalg.norm(np.diff(dc, axis=0), axis=1).sum())
    d_src = np.asarray(h["d_src"], np.float64)
    n_win = min(4, T_l - 1)                                                                    # first B.5 window ~ 12 source frames = 3 latent steps
    rec["object_path"] = {"channel": channel, "drift_model": lock["trajectory"]["drift_model_branch" if channel == "branch" else "drift_model_x0"],
                          "drift_note": "x0: B.5's track as delivered (Stage B drift fit upstream); g_t not applied", "path_len": path,
                          "net_displacement": float(np.linalg.norm(dc[-1])),
                          "vipe_vs_d4rt_disp_diff_over_path": float(np.linalg.norm(dc - dc_d4rt, axis=1).max() / max(path, 1e-12)),
                          "heading_vs_d_src_deg_first_window": angle_deg(dc[n_win], d_src) if np.linalg.norm(dc[n_win]) > 0 else None,
                          "heading_window_latent_steps": int(n_win)}

    # ---- primitives over time: static background; object = G_tar (re-gauged about the camera centre if the rule says so), R_t = I
    Vo = voxel_downsample(contact.rescale_about_camera(V_obj, s_c), float(lock["lift"]["obj_voxel_frac"]))
    c0 = contact.rescale_about_camera(V_obj, s_c).mean(0)
    c_t = c0 + dc
    Xo_t = project.move_object(Vo, np.tile(np.eye(3), (T_l, 1, 1)), c0, c_t)
    X_all = np.concatenate([np.broadcast_to(X_bg, (T_l,) + X_bg.shape), Xo_t], 1)
    is_obj = np.arange(X_all.shape[1]) >= len(X_bg)
    pbar, dep = project.project(X_all, E_l, Kp)
    vis = project.visibility(pbar, dep, Hp, Wp, float(lock["rasterizer"]["zbuf_rel_tol"]))
    O_gtar = np.stack([rasterize.occupancy(pbar[t], vis[t], is_obj, Hp, Wp) for t in range(T_l)])
    # occupancy of record (reviewer 2026-10-06g): the PAINTED mask warped by G_tar's low-pass displacement field (affine fit per frame
    # to the visible object primitives' patch displacements); G_tar keeps the z-buffer, F_geo and T_obj.
    with np.load(lp) as z:
        am = z["anchor_mask_sq"].astype(bool)
    Hq, Wq = am.shape
    st = int(m3cfg["occupancy"]["warp_stride_px"])
    vv, uu = np.mgrid[st // 2:Hq:st, st // 2:Wq:st]
    sel = am[vv, uu]
    Wn_ = int(lat["Wp"]) * int(lat["px_per_patch"]); Hn_ = int(lat["Hp"]) * int(lat["px_per_patch"])
    pm0 = np.stack([(uu[sel] + 0.5) * (Wn_ / Wq) / int(lat["px_per_patch"]), (vv[sel] + 0.5) / int(lat["px_per_patch"])], 1)   # painted pixels, patch units
    O = np.zeros((T_l, Hp, Wp)); warp_fit = []
    for t in range(T_l):
        ok = is_obj & vis[0] & vis[t] & np.isfinite(pbar[0]).all(-1) & np.isfinite(pbar[t]).all(-1)
        O[t], res = rasterize.warp_mask_occupancy(pm0, pbar[0][ok], pbar[t][ok], Hp, Wp)
        warp_fit.append(res)
    rec["occupancy"] = {"source": m3cfg["occupancy"]["source"], "affine_warp_residual_patches_median": [None if v is None else round(v, 3) for v in warp_fit],
                        "painted_vs_gtar_patches_t0": [int((O[0] > 0.5).sum()), int((O_gtar[0] > 0.5).sum())],
                        "iou_painted_vs_gtar_t0": float(((O[0] > 0.5) & (O_gtar[0] > 0.5)).sum() / max(((O[0] > 0.5) | (O_gtar[0] > 0.5)).sum(), 1)),
                        "centroid_target_patches": [[float(a), float(b)] for a, b in (pm0.mean(0) + (project.project(c_t[k][None], E_l[k:k + 1], Kp)[0][0, 0] - project.project(c_t[0][None], E_l[:1], Kp)[0][0, 0]) for k in range(T_l))]}
    F0 = np.zeros((T_l, Hp, Wp, 2)); C0 = np.zeros((T_l, Hp, Wp))
    for t in range(T_l):
        F0[t], C0[t] = rasterize.pair_fields(pbar[0], pbar[t], vis[0] & ~is_obj, vis[t] & ~is_obj, Hp, Wp, EPS)
    cov = rasterize.coverage_curve(C0, O, EPS)
    w_ns = (1.0 - O) * (~sky)[None]                                            # reviewer 2026-10-06b: sky patches never hold a frame-0 primitive
    cov_ns = (C0 * w_ns).sum((1, 2)) / (w_ns.sum((1, 2)) + EPS)
    anchors = rasterize.anchor_schedule(cov, float(lock["anchors"]["tau_anchor"]), int(lock["anchors"]["max_anchors"]))
    anchors_ns = rasterize.anchor_schedule(cov_ns, float(lock["anchors"]["tau_anchor"]), int(lock["anchors"]["max_anchors"]))
    excl = bool(__import__("yaml").safe_load(Path("configs/m3.yaml").read_text(encoding="utf-8"))["m3"]["coverage"]["exclude_sky"])
    anchors_rec, cov_rec = (anchors_ns, cov_ns) if excl else (anchors, cov)          # the schedule of record (owner decision in configs/m3.yaml)
    mag = np.linalg.norm(F0, axis=-1)
    rec["coverage"] = {"cov_curve": [round(float(v), 4) for v in cov], "cov_curve_sky_excluded": [round(float(v), 4) for v in cov_ns],
                       "anchors_of_record": anchors_rec, "schedule_excludes_sky": excl, "anchors": anchors, "anchors_sky_excluded": anchors_ns, "sky_patch_frac": float(sky.mean()),
                       "bg_flow_median_patches_last": float(np.median(mag[-1][C0[-1] > 0.5])) if (C0[-1] > 0.5).any() else None,
                       "bg_flow_p95_patches_last": float(np.quantile(mag[-1][C0[-1] > 0.5], 0.95)) if (C0[-1] > 0.5).any() else None,
                       "bg_flow_median_patches_step1": float(np.median(mag[1][C0[1] > 0.5])) if T_l > 1 and (C0[1] > 0.5).any() else None,
                       "obj_patches_t0": int((O[0] > 0.5).sum()), "obj_patches_last": int((O[-1] > 0.5).sum()),
                       "obj_centre_shift_patches": float(np.linalg.norm(
                           project.project(c_t[-1][None], E_l[-1:], Kp)[0][0, 0] - project.project(c_t[0][None], E_l[:1], Kp)[0][0, 0]))}
    assert np.isfinite(F0).all() and np.isfinite(cov).all()
    tag = ("" if regauge else "_noregauge") + ("" if bridge_run == "rec" else f"_beta_{bridge_run}")
    gp = lp.parent / f"geometry{tag}.npz"
    if gp.exists():
        with np.load(gp) as zprev:
            prev_src = str(zprev["object_path_source"]) if "object_path_source" in zprev.files else "b5_stitched_v1"
        if prev_src != "c5_same_frame":
            arch = lp.parent / f"geometry{tag}_{prev_src}.npz"                      # rule 24: the previous path's geometry is kept beside the new one
            if not arch.exists():
                import shutil; shutil.copy2(gp, arch); print(f"[sint2] archived the previous geometry as {arch.name}", flush=True)
    np.savez_compressed(lp.parent / f"geometry{tag}.npz", E_l=E_l, E_full=E_full, c_tar=c_t, R_t=np.tile(np.eye(3), (T_l, 1, 1)), anchors=np.array(anchors_rec), anchors_sky_included=np.array(anchors),
                        cov_curve=cov, cov_curve_sky_excluded=cov_ns, O=O.astype(np.float32), O_gtar=O_gtar.astype(np.float32), c_tar_raw=(Xw_raw - Xw_raw[0]) + c0, Fgeo0=F0.astype(np.float32), Cgeo0=C0.astype(np.float32),
                        Kp=Kp, s_c=np.float64(s_c), s_delta=np.float64(s_delta), beta=np.float64(beta if beta is not None else np.nan), bridge_run=str(bridge_run), object_path_source="c5_same_frame", c_tar_full_raw=Xw_full.astype(np.float64), c_tar_full_smooth=Xw_sm_full.astype(np.float64),
                        smoothing_applied=bool(smooth_ok), src_frames=src_frames, sky_lattice=sky, V_obj_prims=Vo.astype(np.float32),
                        F_pair=np.int64(F_pair), T_l=np.int64(T_l))
    (lp.parent / f"sint_step2{tag}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", default=__import__("os").environ.get("MT_ROOT", "/content/drive/MyDrive/motion_transfer"))
    ap.add_argument("--no-regauge", action="store_true", help="the un-re-gauged sub-row: s_c forced to 1")
    ap.add_argument("--pairs", nargs="*", default=None, help="clip/category to run (default: every pair with a lift)")
    ap.add_argument("--camera", default="vipe_a5", choices=["vipe_a5", "camera_c_ba"], help="camera_c_ba: the census camera of record (camera C + BA, source gauge, no bridge) for the experiment set; vipe_a5: the six pairs' record")
    ap.add_argument("--bridge", default="rec", choices=["rec", "mid", "far"], help="v2.6 §2.3: which bridge value the geometry cache is built for (rec = every ablation; far = the full-method repeat; mid = the optional third run)")
    args = ap.parse_args(argv)
    lock = H.load_m3_lock()
    assert args.camera == "camera_c_ba" or lock["trajectory"]["source"] == "vipe_a5", "only the A.5 ViPE trajectory is implemented (D-M3-07 (a))"
    root = Path(args.drive_root)
    recs = []
    for lp in sorted((root / "cache" / "m3").glob("*/*/lift.npz")):
        if args.pairs and f"{lp.parent.parent.name}/{lp.parent.name}" not in args.pairs:
            continue
        try:
            recs.append(run_pair(root, lp, lock, regauge=not args.no_regauge, bridge_run=args.bridge, camera_src=args.camera))
        except Exception as e:                       # reported per pair, never swallowed
            recs.append({"pair": f"{lp.parent.parent.name}/{lp.parent.name}", "error": f"{type(e).__name__}: {e}"})
        print(f"[m3 S-int 2] {recs[-1]['pair']}: {recs[-1].get('error', 'done')}")
    tag = ("_noregauge" if args.no_regauge else "") + ("" if args.bridge == "rec" else f"_beta_{args.bridge}")
    rep = root / "reports" / "m3_sint"; rep.mkdir(parents=True, exist_ok=True)
    jf = rep / f"sint_step2{tag}.json"                                                # rule 24: one entry per pair, extended across per-pair invocations
    prev = json.loads(jf.read_text(encoding="utf-8")) if jf.exists() else []
    done_ = {r["pair"] for r in recs}
    recs_all = sorted([r for r in prev if r.get("pair") not in done_] + recs, key=lambda r: r["pair"])
    jf.write_text(json.dumps(recs_all, indent=1), encoding="utf-8")
    f = lambda v, n=3: "-" if v is None else f"{v:.{n}f}"          # noqa: E731
    md = [f"# Module 3 S-int step 2 — camera path, object path, coverage, anchors{' (UN-RE-GAUGED sub-row)' if args.no_regauge else ''}", "",
          "R_t = I on every pair (rotation source not yet chosen). Coverage is the share of non-object patches in frame t that the frame-0 scene still explains.", "",
          "| pair | ch | T_l | s_c | cam path / rot° | obj path | ViPE-vs-D4RT disp diff / path | heading vs d_src ° | cov t=1 / mid / last | same, sky excluded | anchors (sky excl.) | bg flow last: median / p95 patches | obj centre shift patches |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs_all:
        if "error" in r:
            md.append(f"| {r['pair'][-18:]} | ERROR {r['error'][:200]} |"); continue
        c, o, cam = r["coverage"], r["object_path"], r["camera"]
        cv, cn = c["cov_curve"], c["cov_curve_sky_excluded"]; m_ = len(cv) // 2
        md.append(f"| {r['pair'][-18:]} | {o['channel']} | {r['T_l']} | {f(r['contact']['s_c_applied'])} | {f(cam['cam_path_len'])} / {f(cam['cam_rot_total_deg'], 1)} | {f(o['path_len'])} | "
                  f"{f(o['vipe_vs_d4rt_disp_diff_over_path'])} | {f(o['heading_vs_d_src_deg_first_window'], 1)} | {f(cv[1] if len(cv) > 1 else None, 2)} / {f(cv[m_], 2)} / {f(cv[-1], 2)} | "
                  f"{f(cn[1] if len(cn) > 1 else None, 2)} / {f(cn[m_], 2)} / {f(cn[-1], 2)} | {c['anchors']} ({c['anchors_sky_excluded']}) | "
                  f"{f(c['bg_flow_median_patches_last'], 2)} / {f(c['bg_flow_p95_patches_last'], 2)} | {f(c['obj_centre_shift_patches'], 2)} |")
    md += ["", "## Decisions needed from John (S-int hard stop)", "",
           "1. Review per pair: is the object path plausible (heading vs d_src; ViPE-vs-D4RT difference), and does the coverage curve make sense for the camera motion?",
           "2. Sky: anchors computed with and without sky patches in the denominator (reviewer's v2.6 item); choose which is the schedule of record.",
           "3. Any pair with an ERROR row is not carried to the next stage until its cause is read."]
    (rep / f"sint_step2{tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
