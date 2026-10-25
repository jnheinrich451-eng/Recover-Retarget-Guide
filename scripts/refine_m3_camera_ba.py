#!/usr/bin/env python
"""Module 3 — camera C refined by bundle adjustment (reviewer 2026-10-07b). CPU, READ-ONLY on every cache; writes
cache/m3/<clip>/camera_c_ba.npz and reports/m3_sint/camera_c_ba.{md,json}. Gate as camera C: held-out static tracks lifted
at t_a, reprojected through the refined cameras, median px <= lock.trajectory.camera_c_gate_patch over the full window.

Pre-registered expectation (reviewer, reversed from the night before): the BA-refined camera passes the 1-patch gate on 035
over the full window, because with baseline/depth ~ 1 the points triangulate superbly and BA fixes what a single-frame lift
cannot (lifted-depth error amplified by baseline, the measured cause of camera C's rising error on 035).

    python scripts/refine_m3_camera_ba.py                # all clips with a camera_c.npz
    python scripts/refine_m3_camera_ba.py --clips 000000000035.1.003
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.check_m3_camera import in_mask, reproj_px, rot_angle_deg       # noqa: E402
from src.generate.m3 import bundle_adjust as BA, handoff as H                # noqa: E402
from src.generate.m3.geometry import camera, vipe_camera                     # noqa: E402
from src.preprocess.stage_c import finalize                                  # noqa: E402
from src.anchor.m2e import alpha as m2e_alpha                                # noqa: E402  (the ground band's definition, as in run_m3_sint)
from src.generate.m3 import contact as m3_contact                            # noqa: E402  (base_region: the object's bottom-edge rows)


def ground_band_256(root: Path, clip: str, cat: str, h: dict, M_ta: np.ndarray, side: int = 256, census: bool = False) -> np.ndarray:
    """The m2_e ground band (annulus around the object, below the horizon, valid, not the source hole) at the 1024^2 anchor frame,
    resampled to the Stage-A grid. Cached in cache/m3/<clip>/ground_band_256.npz (idempotent). census=True reads the SOURCE-side band
    written by scripts/run_census.py (cache/census/<clip>/ground_band_256.npz); it is never built here and never mixed with the anchor's."""
    import cv2
    if census:
        cf = root / "cache" / "census" / clip / "ground_band_256.npz"
        assert cf.exists(), f"{cf} missing: run scripts/run_census.py for {clip} first"
        with np.load(cf) as z:
            return z["band"].astype(bool)
    cf = root / "cache" / "m3" / clip / "ground_band_256.npz"
    if cf.exists():
        with np.load(cf) as z:
            return z["band"].astype(bool)
    cache = root / "cache"
    qd = json.loads((cache / "m2_d" / clip / cat / "qc_m2d.json").read_text(encoding="utf-8"))
    with np.load(cache / "m2_e" / clip / cat / "anchor_masks.npz") as z:
        amask = z[f"mask_seed_{qd['selected_seed']}"].astype(bool)
    with np.load(cache / "m2_c" / clip / cat / "control.npz") as z:
        M_obj, valid_c = z["M_obj"].astype(bool), z["valid"].astype(bool)
    hole = cv2.resize(M_ta.astype(np.uint8), (1024, 1024), interpolation=cv2.INTER_NEAREST).astype(bool)
    acfg = __import__("yaml").safe_load(Path("configs/m2e.yaml").read_text(encoding="utf-8"))["m2e"]["alpha"]
    band, _ = m2e_alpha.ground_band(amask, M_obj, np.asarray(h["K_1024"], np.float64), np.asarray(h["n_plane"], float), valid_c, hole,
                                    int(acfg["band_inner_px"]), float(acfg["band_outer_frac"]))
    band_s = cv2.resize(band.astype(np.float32), (side, side), interpolation=cv2.INTER_AREA) > 0.5
    cf.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cf, band=band_s, definition="m2_e ground_band at anchor 1024^2, INTER_AREA to 256^2, > 0.5")
    return band_s

PRIOR_WEIGHT_PX = 200.0       # pixels of residual per unit relative depth error on the anchor points. 200 = the sweep's setting at which the
                              # x0 clips stop degrading (camera_c_ba_sweep, 2026-10-07); the reviewer names the full-set BA at this prior as
                              # the admitted camera (2026-10-07h). 50 was the first run's value.
N_ANCHOR = 40                 # frame-t_a points whose lifted depth fixes the scale (a prior, not a constraint on the rest)
MIN_OBS_PER_PT = 6
# Condition 2 (reviewer 2026-10-07h/i): the negative controls are the two REGISTERED pairs, the cameras the lift column rejected. ViPE/sigma_hat
# on the x0 clips is the camera of record and is supposed to pass; applying the rule to every clip would declare "no power" on a correct result.
REGISTERED_CONTROLS = {"000000000035.1.003": "vipe_over_sigma", "000000000265.2.005": "d4rt_raw_G"}


def run_clip(root: Path, clip: str, cat: str, lock: dict, prior_weight: float, n_anchor: int, init_px_max: float, sweep_mode: bool = False, far_half: bool = False,
             min_points: int = 30, exclude_mask=None, free_focal: bool = False, census: bool = False) -> dict:
    h = json.loads((root / "cache" / ("census" if census else "m2_e") / clip / cat / "handoff.json").read_text(encoding="utf-8"))
    t_a = int(h["t_a"]); F_pair, _ = camera.frames_for_pair(h["n_frames_after_t_a"])
    gate_px = float(lock["trajectory"]["camera_c_gate_patch"]) * 256.0 / int(lock["lattice"]["Wp"])
    gauge_tol = lock["trajectory"].get("triangulated_gauge_tol")
    assert gauge_tol is not None, "lock m3.trajectory.triangulated_gauge_tol missing (John adds; reviewer 2026-10-07h proposed 0.05)"
    gauge_tol = float(gauge_tol)
    # the triangulated column has the lift error removed, so it carries its own threshold (reviewer 2026-10-07j: 0.3 patch = the tracker floor,
    # the precision at which motion is known on this lattice; not derived from the cameras it grades; recalibrate on the Chapter 5 set)
    tri_patch = lock["trajectory"].get("triangulated_gate_patch")
    assert tri_patch is not None, "lock m3.trajectory.triangulated_gate_patch missing (John adds; reviewer 2026-10-07j set 0.3 patch)"
    tri_px = float(tri_patch) * 256.0 / int(lock["lattice"]["Wp"])
    with np.load(root / "cache" / "stage_a" / clip / "bundle.npz") as z:
        G, K, Xl, P, Q = (z[k].astype(np.float64) if k != "Q" else z[k].astype(bool) for k in ("G", "K", "X_local", "P", "Q"))
    with np.load(root / "cache" / "stage_b" / f"{clip}.npz") as z:
        y_dyn = z["y_dyn"].astype(bool)
    M, _, _ = finalize.read_mask(root / "cache" / "stage_c" / clip)
    cc = np.load(root / "cache" / "m3" / clip / "camera_c.npz")
    E_init, solved = cc["E_c"].astype(np.float64), cc["solved"].astype(bool)
    cc_rec = json.loads((root / "reports" / ("census" if census else "m3_sint") / "camera_c.json").read_text(encoding="utf-8"))
    cc_held_median = next(r["heldout_reproj_px"]["median_all"] for r in cc_rec if r["pair"].startswith(clip))
    assert solved.all(), "camera C must be solved on every frame before refinement"
    static = ~y_dyn
    for t in range(t_a, t_a + F_pair):
        static &= ~(Q[:, t] & in_mask(P[:, t], M[t]))
    n_static_before_excl = int(static.sum())
    if exclude_mask is not None:                                              # reviewer 2026-10-07j: a region the human names (e.g. water), at t_a
        assert exclude_mask.shape == M[t_a].shape, (exclude_mask.shape, M[t_a].shape)
        static &= ~(Q[:, t_a] & in_mask(P[:, t_a], exclude_mask))
    idx = np.nonzero(static)[0]
    solve, held = idx[idx % 2 == 0], idx[idx % 2 == 1]
    # initial world points: solving tracks lifted at t_a (world = camera-t_a); only tracks valid at t_a with enough observations
    win = np.arange(t_a, t_a + F_pair)
    n_obs = Q[solve][:, win].sum(1)
    use = solve[(Q[solve, t_a]) & (n_obs >= MIN_OBS_PER_PT)]
    if far_half:
        zmed = float(np.median(Xl[use, t_a, 2])); use = use[Xl[use, t_a, 2] >= zmed]
    X0 = Xl[use, t_a]
    # Observation assembly (reviewer 2026-10-07c, after the 2e6-px initial residual on 035): keep an observation only if the
    # t_a-lifted point, moved by camera C's initial pose, is in front of the camera, inside the frame, and within `init_px_max`
    # of where the tracker saw it. Observations kept after a point had left the view were the bug.
    oc, op, ouv, dropped = [], [], [], {"behind": 0, "outside": 0, "far_from_init": 0}
    for j, i in enumerate(use):
        for k, t in enumerate(win):
            if not Q[i, t]:
                continue
            Xc = X0[j] @ E_init[k, :3, :3].T + E_init[k, :3, 3]
            if Xc[2] <= 0.05:
                dropped["behind"] += 1; continue
            uv0 = reproj_px(X0[j][None], E_init[k], K[t])[0]
            if not (0 <= uv0[0] < 256 and 0 <= uv0[1] < 256):
                dropped["outside"] += 1; continue
            if np.linalg.norm(uv0 - P[i, t]) > init_px_max:
                dropped["far_from_init"] += 1; continue
            oc.append(k); op.append(j); ouv.append(P[i, t])
    oc, op, ouv = np.array(oc), np.array(op), np.array(ouv, np.float64)
    keep_pts = set(np.unique(op).tolist()); assert len(keep_pts) >= 20, "too few points survive the assembly filter"
    order = np.argsort(np.abs(X0[:, 2] - np.median(X0[:, 2])))
    anchor = np.array([a for a in order if a in keep_pts][: min(n_anchor, len(keep_pts))])
    E_ba, X_ba, info = BA.bundle_adjust(E_init, X0, oc, op, ouv, K[win], anchor, prior_weight, free_focal=free_focal)
    K_used = K[win].copy()
    if free_focal:                                                           # the refined cameras are read with the focal they were fitted with
        K_used[:, 0, 0] *= info["focal_scale"]; K_used[:, 1, 1] *= info["focal_scale"]
    info["dropped_observations"] = dropped
    # precondition (reviewer): the initial residual must be consistent with camera C's own held-out error, else the assembly is wrong
    init_ok = info["reproj_rms_before"] <= 2.0 * float(cc_held_median) + 1.0
    info["init_consistent_with_camera_c"] = bool(init_ok)
    # held-out gate: same instrument as camera C (lift at t_a, reproject), now through the refined cameras
    held_ok = held[Q[held, t_a]]; Xh = Xl[held_ok, t_a]
    errs, errs_init = [], []
    for k in range(1, F_pair):
        t = t_a + k; v = Q[held_ok, t] & ~in_mask(P[held_ok, t], M[t])
        if v.sum() < 12:
            errs.append(np.nan); errs_init.append(np.nan); continue
        errs.append(float(np.nanmedian(np.linalg.norm(reproj_px(Xh[v], E_ba[k], K_used[k]) - P[held_ok[v], t], axis=1))))
        errs_init.append(float(np.nanmedian(np.linalg.norm(reproj_px(Xh[v], E_init[k], K[t]) - P[held_ok[v], t], axis=1))))
    errs, errs_init = np.array(errs), np.array(errs_init)
    # The lift-free read of the cameras (reviewer 2026-10-07h: gate of record for every clip and every camera; the lift column carries the
    # held-out points' own single-frame depth error amplified by baseline, f T dZ / Z^2, which Module 3 never consumes). Three conditions,
    # fixed before any 035 number on this column is read: (1) split in time, (2) negative controls must FAIL it, (3) gauge check.
    held_all = held                                                            # every held-out static track (triangulation needs no t_a lift)
    P_w, Q_w = P[held_all][:, win], Q[held_all][:, win]
    excl_w = np.stack([in_mask(P[held_all, t], M[t]) for t in win], 1)
    Z_lift = np.where(Q[held_all, t_a], Xl[held_all, t_a, 2], np.nan)
    X_lift_w = np.where(Q[held_all, t_a][:, None], Xl[held_all, t_a], np.nan)
    tri = BA.triangulated_heldout(E_ba, K_used, P_w, Q_w, excl_w, Z_lift, tri_px, gauge_tol, min_points=min_points, return_points=True, X_lift=X_lift_w)
    # the gauge as a measurement against the ground band (reviewer 2026-10-07k); the lifted-depth terciles stay as information
    band = ground_band_256(root, clip, cat, h, M[t_a], census=census)
    band_g = BA.band_gauge(tri["points"], band, gauge_tol, min_points=min_points)
    rebin = BA.rebin_tables(tri["points"])
    # reviewer 2026-10-07l: (1) the contact bridge beta at the object's bottom-edge rays, (2) band shape, (3) object bias over time, (4) band parallax
    contact_uv = BA.bottom_edge_rays(M[t_a], 0.10)                       # per-column bottom edge within the lowest 10 % of the object (reviewer 2026-10-07m)
    cbr = BA.contact_bridge(tri["points"], band, K_used[0], contact_uv, band_g["ratio"], gauge_tol)
    cbr["within_band_ci_halfwidth"] = (None if (cbr.get("beta_over_rho_minus_1") is None or band_g.get("ci_half_width") is None)
                                       else bool(abs(cbr["beta_over_rho_minus_1"]) <= band_g["ci_half_width"]))
    dp = root / "cache" / "m3" / clip / cat / "contact_decision.json"
    s_c = float(json.loads(dp.read_text(encoding="utf-8"))["s_c"]) if dp.exists() else None
    cbr["s_c"] = s_c; cbr["s_c_times_rho"] = None if (s_c is None or not band_g["ratio"]) else float(s_c * band_g["ratio"])
    uv_in_band = np.zeros(len(held_all), bool)
    for p in tri["points"]:
        if p["uv_ta"] is not None:
            ui, vi = int(round(p["uv_ta"][0])), int(round(p["uv_ta"][1]))
            if 0 <= ui < band.shape[1] and 0 <= vi < band.shape[0] and band[vi, ui]:
                uv_in_band[p["track"]] = True
    cbr["band_parallax_px"] = BA.band_parallax(P_w, Q_w, excl_w, uv_in_band)
    frames_o = [t_a + k for k in (0, 20, 40, 60, 80) if k < F_pair]
    # object tracks = sparse (y_dyn) UNION Stage C.5 dense (X_local_dense, P_dense, Q_dense; all object): the sparse base had 5-19 tracks per frame
    c5 = root / "cache" / "stage_c5" / f"{clip}.npz"
    if c5.exists():
        with np.load(c5) as z:
            Xl_o, P_o, Q_o = np.concatenate([Xl, z["X_local_dense"].astype(np.float64)]), np.concatenate([P, z["P_dense"].astype(np.float64)]), np.concatenate([Q, z["Q_dense"].astype(bool)])
            y_o = np.concatenate([y_dyn, np.ones(len(z["Q_dense"]), bool)])
        n_dense = int(len(Q_o) - len(Q))
    else:
        Xl_o, P_o, Q_o, y_o, n_dense = Xl, P, Q, y_dyn, 0
    rho_o = BA.object_bias_over_time(Xl_o, P_o, Q_o, y_o, M, K, frames_o, m3_contact.base_region, float(lock["int"]["contact_rows"]))
    rho_o["n_dense_tracks_added"] = n_dense
    # residual by image third (reviewer 2026-10-07l: the masked rerun must report the bottom-left / bottom-centre cells beside the ratio)
    uv_all = np.array([p["uv_ta"] if p["uv_ta"] is not None else p["uv_first"] for p in tri["points"]], np.float64).reshape(-1, 2)
    err_all = np.array([p["err_median"] for p in tri["points"]])
    r3, c3 = np.digitize(uv_all[:, 1], [256 / 3, 512 / 3]), np.digitize(uv_all[:, 0], [256 / 3, 512 / 3])
    by_region = {f"{rn}/{cn}": {"median_px": float(np.median(err_all[(r3 == i) & (c3 == j)])) if ((r3 == i) & (c3 == j)).sum() >= 5 else None, "n": int(((r3 == i) & (c3 == j)).sum())}
                 for i, rn in enumerate(("top", "middle", "bottom")) for j, cn in enumerate(("left", "centre", "right"))}
    pts_rec = tri.pop("points")
    # the retired lifted-depth tercile gate: kept as information only, without an outcome (reviewer 2026-10-07l item 5)
    tri["reprojection_outcome"] = "pass" if tri["reproj_ok"] else "fail"; tri.pop("outcome", None)
    tri["lift_tercile_information"] = {k: v for k, v in tri.pop("gauge").items() if k not in ("outcome", "ok", "read_on", "near")}
    # (2) negative controls: the cameras the lift column already rejected. Registered: ViPE/sigma_hat on 035 (16.5 px lift) and raw D4RT G
    # across the window boundaries on 265.2.005 (12-14.5 px lift). If either PASSES this column, the column has no power.
    Gi = np.linalg.inv(G); E_d = np.stack([Gi[t_a + k] @ G[t_a] for k in range(F_pair)])
    controls = {"d4rt_raw_G": BA.triangulated_heldout(E_d, K[win], P_w, Q_w, excl_w, Z_lift, tri_px, gauge_tol, min_points=min_points)}
    pf = sorted((root / "bakeoff" / "stage_a5" / clip / "vipe" / "pose").glob("*.npz"))
    crec_p = root / lock["trajectory"]["sigma_t_record"]
    if pf and crec_p.exists() and clip in json.loads(crec_p.read_text(encoding="utf-8")):
        with np.load(pf[0]) as z:
            P_c2w = z["data"].astype(np.float64)
        sig_ta = float(json.loads(crec_p.read_text(encoding="utf-8"))[clip]["sigma_abs"][t_a])
        E_s, _ = vipe_camera.phase2_camera(P_c2w, t_a, sig_ta, F_pair, 1e-4)
        controls["vipe_over_sigma"] = BA.triangulated_heldout(E_s, K[win], P_w, Q_w, excl_w, Z_lift, tri_px, gauge_tol, min_points=min_points)
    else:
        controls["vipe_over_sigma"] = None
    # a control "passes the column" when it passes the whole record gate (reprojection AND gauge); reprojection alone is reported too
    controls_fail = {k: (None if v is None else (v["outcome"] != "pass")) for k, v in controls.items()}
    registered = REGISTERED_CONTROLS.get(clip)
    if registered is None:
        power = "not a condition on this clip (controls informational)"; power_ok = True
    elif controls.get(registered) is None:
        power = f"registered control {registered} unavailable"; power_ok = False
    else:
        power_ok = bool(controls_fail[registered]); power = f"registered control {registered}: {'fails (column has power)' if power_ok else 'PASSES (column has no power)'}"
    # admission (reviewer 2026-10-07k): time-split reprojection <= the column's threshold, the registered control failing, band ratio determined
    record = ("not a result: init inconsistent (assembly)" if not init_ok else
              "no power: the registered negative control passed the column" if not power_ok else
              "fail (reprojection)" if not tri["reproj_ok"] else
              "pass" if band_g["outcome"] == "determined" else f"undetermined (band gauge: {band_g['outcome']})")
    scale_change = float(np.median(X_ba[anchor, 2] / X0[anchor, 2]))
    med = float(np.nanmedian(errs))
    rec = {"pair": f"{clip}/{cat}", "t_a": t_a, "F_pair": F_pair, "ba": info, "n_solve_points": int(len(use)), "n_held": int(len(held_ok)),
           "heldout_px": {"median_all": med, "p90": float(np.nanquantile(errs, 0.9)), "curve_every8": [round(float(v), 2) for v in errs[::8]],
                          "median_all_before_ba": float(np.nanmedian(errs_init))},
           "heldout_triangulated": tri, "band_gauge": band_g, "rebin": rebin, "gauge_n_points_of_scored": [int(tri["lift_tercile_information"]["n"]), int(tri["n_points"])],
           "contact_bridge": cbr, "object_bias_over_time": rho_o, "residual_by_region": by_region,
           "negative_controls": controls, "negative_controls_fail": controls_fail, "registered_control": registered, "power": power,
           "free_focal": {"enabled": free_focal, "focal_scale": info.get("focal_scale")},
           "gate_of_record": {"column": "triangulated split (reviewer 2026-10-07h)", "px_256": tri_px, "patch": float(tri_patch), "gauge_tol": gauge_tol, "outcome": record},
           "static_set": {"n_static": int(static.sum()), "n_before_exclude_mask": n_static_before_excl, "exclude_mask": exclude_mask is not None},
           "anchor_depth_scale_change": scale_change, "rot_change_vs_init_deg_max": float(max(rot_angle_deg(E_ba[k, :3, :3], E_init[k, :3, :3]) for k in range(F_pair))),
           "gate": {"px_256": gate_px, "outcome": ("not a result: init inconsistent (assembly)" if not init_ok else ("pass" if med <= gate_px else "fail"))},
           "x0_no_degradation": bool(med <= float(np.nanmedian(errs_init)) + 0.1),
           "instrument": {"prior_weight_px": prior_weight, "n_anchor": int(len(anchor)), "init_px_max": init_px_max, "far_half": far_half, "loss": "linear then huber (3xMAD)", "split": "track index parity"}}
    if not sweep_mode:
        np.savez_compressed(root / "cache" / "m3" / clip / "camera_c_ba.npz", E_c=E_ba, X_static=X_ba, t_a=np.int64(t_a), F_pair=np.int64(F_pair))
        np.savez_compressed(root / "cache" / "m3" / clip / "camera_c_ba_heldout.npz", points=json.dumps(pts_rec), band_gauge=json.dumps(band_g))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", default=__import__("os").environ.get("MT_ROOT", "/content/drive/MyDrive/motion_transfer"))
    ap.add_argument("--clips", nargs="+", default=None)
    ap.add_argument("--census", action="store_true", help="census mode: source-side handoff and band (cache/census), reports/census/, record extended per clip")
    ap.add_argument("--prior-weight", type=float, default=PRIOR_WEIGHT_PX)
    ap.add_argument("--n-anchor", type=int, default=N_ANCHOR)
    ap.add_argument("--init-px-max", type=float, default=12.0, help="assembly: drop observations farther than this from the initial projection")
    ap.add_argument("--far-half", action="store_true", help="solve on the FAR half of the static points only (reviewer 2026-10-07c: track contamination on a follow-cam by near ground/shadow/dust)")
    ap.add_argument("--exclude-mask", default=None, help="256x256 binary PNG; static tracks inside it at t_a are removed from BOTH the solve and the held-out sets "
                    "(reviewer 2026-10-07j: the water on 035). Output tag _masked; no cache written.")
    ap.add_argument("--free-focal", action="store_true", help="Stage-A confirming test (reviewer 2026-10-07k): BA with one shared focal scale freed; "
                    "report-only (tag _freefocal). Reading: 035's residual collapses and the focal moves by several percent while 404 stays within ~1 %% -> intrinsics.")
    ap.add_argument("--sweep", action="store_true", help="instrument calibration (report only, no cache written): prior weight x anchor set; "
                    "criterion pre-registered by the reviewer: the x0 clips held-out error must not degrade")
    args = ap.parse_args(argv)
    lock = H.load_m3_lock(); root = Path(args.drive_root)
    m3cfg = __import__("yaml").safe_load(Path("configs/m3.yaml").read_text(encoding="utf-8"))["m3"]
    min_points = int(m3cfg["trajectory"]["triangulated_min_points"])
    rows, seen = [], set()
    excl = None
    if args.exclude_mask:
        from PIL import Image
        excl = np.asarray(Image.open(args.exclude_mask).convert("L")) > 127
    for hp in sorted((root / "cache" / ("census" if args.census else "m2_e")).glob("*/*/handoff.json")):
        clip, cat = hp.parent.parent.name, hp.parent.name
        if clip in seen or (args.clips and clip not in args.clips) or not (root / "cache" / "m3" / clip / "camera_c.npz").exists():
            continue
        seen.add(clip)
        grid = [(args.prior_weight, args.n_anchor)] if not args.sweep else [(w, n) for w in (50.0, 200.0, 800.0) for n in (40, 10 ** 6)]
        for w, n in grid:
            try:
                r = run_clip(root, clip, cat, lock, w, n, args.init_px_max, sweep_mode=args.sweep or args.far_half or excl is not None or args.free_focal,
                             far_half=args.far_half, min_points=min_points, exclude_mask=excl, free_focal=args.free_focal, census=args.census)
            except Exception as e:
                r = {"pair": f"{clip}/{cat}", "error": f"{type(e).__name__}: {e}", "instrument": {"prior_weight_px": w, "n_anchor": n}}
            rows.append(r)
            print(f"[m3 camera C + BA] {clip} prior {w} anchors {n}: lift {r.get('error') or r['gate']['outcome']} ({r.get('heldout_px', {}).get('median_all')}); "
                  f"record {r.get('gate_of_record', {}).get('outcome')} ({r.get('heldout_triangulated', {}).get('median_px')})")
    rep = root / "reports" / ("census" if args.census else "m3_sint"); rep.mkdir(parents=True, exist_ok=True)
    tag = "_sweep" if args.sweep else ("_farhalf" if args.far_half else ("_masked" if excl is not None else ("_freefocal" if args.free_focal else "")))
    if args.census and (rep / f"camera_c_ba{tag}.json").exists():                       # census: the record is extended per clip, never replaced
        prev = json.loads((rep / f"camera_c_ba{tag}.json").read_text(encoding="utf-8"))
        rows = [r for r in prev if r["pair"] not in {x["pair"] for x in rows}] + rows
    (rep / f"camera_c_ba{tag}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    f = lambda v, n=2: "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{n}f}"     # noqa: E731
    md = ["# Module 3 — camera C refined by bundle adjustment (held-out gate, same instrument as camera C)", "",
          "| clip | prior px / anchors | points / obs (dropped: behind, outside, far) | reproj RMS before -> after (px) | init consistent? | held-out LIFT px: camera C -> BA (p90) | TRIANGULATED split px (p90; n; dropped) | BAND gauge Z_tri/Z_lift: median [95 % CI] (n; half fit → verify; band reproj px) → outcome | lift-tercile gauge (information): near/mid/far; re-binned by Z_tri near/mid/far; by image row top/middle/bottom | neg. controls (registered = *): D4RT G / ViPE÷σ̂ px → reprojection | x0 no-degradation | anchor-depth scale change | max rot change ° | gate |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            md.append(f"| {r['pair'][:18]} | ERROR {r['error'][:160]} |"); continue
        b, e = r["ba"], r["heldout_px"]
        d = b["dropped_observations"]; ins = r["instrument"]; tr = r["heldout_triangulated"]; g = tr["lift_tercile_information"]
        bg = r["band_gauge"]; rb = r.get("rebin", {})
        tb = lambda key: "/".join(f(v["median"], 3) for v in rb.get(key, {}).values()) or "-"                                           # noqa: E731
        ctl = lambda k: ("-" if r["negative_controls"][k] is None else f"{'*' if r['registered_control'] == k else ''}{f(r['negative_controls'][k]['median_px'])} px → "   # noqa: E731
                         f"{'fails' if not r['negative_controls'][k]['reproj_ok'] else 'PASSES'}")
        md.append(f"| {r['pair'][:18]} | {ins['prior_weight_px']:.0f} / {ins['n_anchor']} | {b['n_points']} / {b['n_obs']} ({d['behind']}, {d['outside']}, {d['far_from_init']}) | {f(b['reproj_rms_before'])} -> {f(b['reproj_rms_after'])} | "
                  f"{b['init_consistent_with_camera_c']} | {f(e['median_all_before_ba'])} -> **{f(e['median_all'])}** ({f(e['p90'])}) | "
                  f"{f(tr['median_px'])} ({f(tr['p90_px'])}; n={tr['n_points']}; {sum(tr['dropped'].values())}) | "
                  f"{f(bg['ratio'], 3)} [{'-' if bg['ci95'] is None else f(bg['ci95'][0], 3) + ', ' + f(bg['ci95'][1], 3)}] (n={bg['n_band_points']}; {f(bg['half_fit'], 3)} → {f(bg['half_verify'], 3)}; {f(bg['band_reproj_median_px'])} px) → **{bg['outcome']}** | "
                  f"{'/'.join(f(v['median'], 3) for v in g['terciles'].values()) or '-'}; {tb('by_triangulated_depth')}; {tb('by_image_row')} | "
                  f"{ctl('d4rt_raw_G')} / {ctl('vipe_over_sigma')} | {r['x0_no_degradation']} | "
                  f"{f(r['anchor_depth_scale_change'], 3)} | {f(r['rot_change_vs_init_deg_max'], 1)}{' | focal x' + f(r['free_focal']['focal_scale'], 4) if r['free_focal']['enabled'] else ''} | lift: {r['gate']['outcome']} (≤ {f(r['gate']['px_256'])} px); **record: {r['gate_of_record']['outcome']}** (≤ {f(r['gate_of_record']['px_256'])} px; band tol ±{r['gate_of_record']['gauge_tol']}) |")
    md += ["", "## Contact bridge, band shape, band parallax, object bias over time (reviewer 2026-10-07l)", "",
           "| clip | band ratio ρ [CI] | β at the contact rays (n rays) | β/ρ − 1 (inside band CI half-width?) | s_c | s_c·ρ | plane tilt ° (tri vs lift) | height ratio |d_tri|/|d_lift| | Spearman ratio vs Z_tri in band | band parallax px: median / p90 (n) | ρ_o(t) at t_a+{0,20,40,60,80}: values; spread | residual px bottom row: left / centre / right (n) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            continue
        c_, bg, ro = r["contact_bridge"], r["band_gauge"], r["object_bias_over_time"]
        bp = c_.get("band_parallax_px", {})
        md.append(f"| {r['pair'][:18]} | {f(bg['ratio'], 3)} [{'-' if bg['ci95'] is None else f(bg['ci95'][0], 3) + ', ' + f(bg['ci95'][1], 3)}] | {f(c_.get('beta'), 3)} ({c_.get('n_contact_rays', 0)}) | "
                  f"{f(c_.get('beta_over_rho_minus_1'), 3)} ({c_.get('within_band_ci_halfwidth')}) | {f(c_.get('s_c'), 3)} | {f(c_.get('s_c_times_rho'), 3)} | {f(c_.get('tilt_deg'), 2)} | {f(c_.get('height_ratio_abs_d'), 3)} | "
                  f"{f(c_.get('spearman_ratio_vs_Ztri_in_band'))} | {f(bp.get('median_px'), 1)} / {f(bp.get('p90_px'), 1)} ({bp.get('n')}) | "
                  + ", ".join(f"{k}: {f(v.get('rho_o'), 3)}" for k, v in ro["per_frame"].items()) + f"; {f(ro['spread_rel'], 3)} | "
                  + " / ".join(f"{f(r['residual_by_region'][f'bottom/{cn}']['median_px'])} ({r['residual_by_region'][f'bottom/{cn}']['n']})" for cn in ("left", "centre", "right")) + " |")
    md += ["", "Expected (reviewer): |β/ρ − 1| inside the band interval's half-width; tilt ≤ 1° and no depth trend (a depth trend would mean a scalar bridge is exact only at the "
           "contact point); ρ_o(t) constant within 5 % (else reported with S4 as a bound on L_obj). β read through the identity s_c·ρ_o = s_c·β: the band bridge applies to the "
           "camera translations only, s_c stays on the object (D-M3-08 amendment text)."]
    md += ["", "Gate of record (reviewer 2026-10-07h): the TRIANGULATED split column, for every clip and every camera, under three conditions fixed before "
           "035's number on it was read: (1) each held-out point triangulated from its even-frame observations and scored on the odd frames, >= 8 observations "
           "spanning half the window; (2) the negative controls (ViPE÷σ̂ on 035, raw D4RT G on 265.2.005) must FAIL the column, else it has no power and the "
           "lift column stays the gate; (3) median Z_tri/Z_lift at t_a within the lock tolerance of 1 (bootstrap interval reported; on 035 each 1 % of scale "
           "is ~0.4 patch of background flow at the last frame). Admitted camera = full-set BA at prior 200; far-half stays a diagnostic. The lift column "
           "stays reported: it measures how sensitive a clip is to depth error, the same amplification MoGe's error meets in the target.",
           "", "Revisions fixed before the 035 number was read (reviewer 2026-10-07i): the gauge is an equivalence claim, so the WHOLE bootstrap interval must "
           "lie inside the tolerance (pass), wholly outside (fail: wrong scale), straddling (undetermined); fewer than `m3.trajectory.triangulated_min_points` "
           "(configs/m3.yaml) scorable points is undetermined; the ratio is given per lifted-depth tercile and, when the terciles differ by more than the "
           "tolerance, the gate is read on the NEAR tercile (where Module 3 takes its scale) and the gauge is marked depth-dependent. Condition 2 applies "
           "only to the registered pairs (ViPE÷σ̂ on 035, raw G on 265.2.005); on the other clips the controls are informational, and ViPE÷σ̂ is the camera "
           "of record there. A control passes the column only when it passes reprojection AND gauge; its reprojection-only number is shown so the two "
           "parts' power can be told apart.",
           "", "Gauge of record (reviewer 2026-10-07k): a MEASUREMENT against the m2_e ground band, where Module 3 takes its scale (k_bg and the object's "
           "contact): median Z_tri/Z_lift over held-out static points inside the band at t_a (selected by image position, free of the lifted-depth "
           "binning artefact); fitted on a random half, verified on the other half within the tolerance; fewer than the minimum count or a bootstrap "
           "half-width above the tolerance is undetermined. A determined ratio is the bridge that replaces sigma_hat (D-M3-08 amendment, John's). "
           "Admission = time-split reprojection within the column's threshold AND the registered control failing AND the band ratio determined. "
           "The lifted-depth terciles are kept as information with the same ratios re-binned by triangulated depth and by image row (a near excess that "
           "survives re-binning is a bias; one that vanishes was selection). The band reprojection residual is reported beside the ratio.",
           "", "Pre-registered (before this version ran): the band ratios on the x0 clips are determined and sit between 1.0 and the near-tercile values; "
           "the near excess survives re-binning by triangulated depth on 404 and 370 (narrow, clean mid/far bins) and is weaker on the two 265 clips."]
    (rep / f"camera_c_ba{tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
