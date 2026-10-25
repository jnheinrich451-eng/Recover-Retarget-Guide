#!/usr/bin/env python
"""Module 3 — camera C: D4RT's camera RE-SOLVED from D4RT's own static tracks, continuous across the encoder-window
boundaries (reviewer 2026-10-06j). CPU, READ-ONLY on every cache; writes cache/m3/<clip>/camera_c.npz (for the clip) and
reports/m3_sint/camera_c.{md,json}. Nothing is bridged, nothing is decided: the gate is reported, John declares.

Procedure (as specified):
  1. static tracks = Stage-A tracks with Stage-B membership excluding the subject (Q valid, y_dyn False, outside the mask);
     split in two halves by track index parity: one half SOLVES, the other is HELD OUT for the gate.
  2. keyframes = t_a and every encoder-window boundary inside the pair's window. At keyframe k the solving tracks are
     lifted with D4RT's own depth at k (X_local[:, k], camera-k coordinates) into the world (= camera-t_a) through the
     ALREADY-SOLVED camera C_k, never through G: that carries the lift's gauge across the boundary.
  3. every frame t: PnP (EPnP + RANSAC, LM refinement) of the current keyframe's world points against the observed track
     pixels at t, under K[t]. Rotation and translation both solved.
  4. gate: held-out half, median reprojection <= lock.trajectory.camera_c_gate_patch (in lattice patches; 256-px scale
     converted with 256 / Wp) over the full window. Jerk of the camera centres reported beside ViPE's.
Sanity rows: camera C vs D4RT's G inside t_a's window (should agree ~ px) and across boundaries; the object path's heading
under camera C vs d_src at 12 frames (should stay near the pure-D4RT 1.7 deg on 035).

    python scripts/check_m3_camera_c.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.check_m3_camera import angle_deg, in_mask, reproj_px, rot_angle_deg      # noqa: E402
from src.generate.m3 import handoff as H                                              # noqa: E402
from src.generate.m3.geometry import camera, vipe_camera                              # noqa: E402
from src.preprocess.stage_c import finalize                                           # noqa: E402

PNP_REPROJ_PX = 4.0          # RANSAC inlier radius at 256 px (instrument setting; recorded in the report, not a gate)
MIN_PTS = 12


def solve_pnp(Xw, uv, K):
    """world points [N,3], pixels [N,2], K [3,3] -> E [4,4] world->camera, inlier count. EPnP + RANSAC, then LM refinement."""
    import cv2
    ok, rvec, tvec, inl = cv2.solvePnPRansac(Xw.astype(np.float64), uv.astype(np.float64), K.astype(np.float64), None,
                                             reprojectionError=PNP_REPROJ_PX, iterationsCount=500, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < MIN_PTS:
        return None, 0
    inl = inl[:, 0]
    rvec, tvec = cv2.solvePnPRefineLM(Xw[inl].astype(np.float64), uv[inl].astype(np.float64), K.astype(np.float64), None, rvec, tvec)
    E = np.eye(4); E[:3, :3] = cv2.Rodrigues(rvec)[0]; E[:3, 3] = tvec[:, 0]
    return E, int(len(inl))


def jerk(E):
    c = np.linalg.inv(E)[:, :3, 3]
    return float(np.median(np.linalg.norm(np.diff(c, n=3, axis=0), axis=1))) if len(c) > 3 else None


def run_clip(root: Path, clip: str, cat: str, lock: dict, census: bool = False) -> dict:
    h = json.loads((root / "cache" / ("census" if census else "m2_e") / clip / cat / "handoff.json").read_text(encoding="utf-8"))
    t_a = int(h["t_a"]); F_pair, T_l = camera.frames_for_pair(h["n_frames_after_t_a"])
    lat = lock["lattice"]; px_per_patch_256 = 256.0 / int(lat["Wp"])
    gate_px = float(lock["trajectory"]["camera_c_gate_patch"]) * px_per_patch_256
    with np.load(root / "cache" / "stage_a" / clip / "bundle.npz") as z:
        G, K, Xl, P, Q = (z[k].astype(np.float64) if k != "Q" else z[k].astype(bool) for k in ("G", "K", "X_local", "P", "Q"))
    qa = json.loads((root / "cache" / "stage_a" / clip / "qc.json").read_text(encoding="utf-8"))
    bounds = [b for b in (qa.get("window_boundaries") or []) if t_a < b < t_a + F_pair]
    with np.load(root / "cache" / "stage_b" / f"{clip}.npz") as z:
        y_dyn = z["y_dyn"].astype(bool)
    M, _, _ = finalize.read_mask(root / "cache" / "stage_c" / clip)
    static = ~y_dyn
    for t in range(t_a, t_a + F_pair):
        static &= ~(Q[:, t] & in_mask(P[:, t], M[t]))                 # never on the subject at any window frame
    idx = np.nonzero(static)[0]
    solve, held = idx[idx % 2 == 0], idx[idx % 2 == 1]
    keyframes = [t_a] + bounds
    E_c = np.full((F_pair, 4, 4), np.nan); E_c[0] = np.eye(4)
    C_k = np.eye(4)                                                    # camera-k -> world for the current keyframe
    Xw, src_ids, log = None, None, []
    for k in range(F_pair):
        t = t_a + k
        if k > 0:                                                   # solve this frame from the CURRENT keyframe's world points
            vis = Q[src_ids, t]
            if vis.sum() >= MIN_PTS:
                E, n_in = solve_pnp(Xw[vis], P[src_ids[vis], t], K[t])
                if E is not None:
                    E_c[k] = E
        if t in keyframes:                                          # then, at a boundary, re-lift through the camera JUST solved
            if t != t_a:
                if not np.isfinite(E_c[k]).all():
                    log.append({"keyframe": int(t), "error": "camera not solved at the boundary frame; chain broken"})
                    break
                C_k = np.linalg.inv(E_c[k])
            ok = solve[Q[solve, t]]
            Xw = Xl[ok, t] @ C_k[:3, :3].T + C_k[:3, 3]; src_ids = ok    # D4RT depth at k, camera-k coordinates, source gauge -> world
            log.append({"keyframe": int(t), "n_lifted": int(len(ok))})
    solved = np.isfinite(E_c).all((1, 2))
    # gate on the HELD-OUT half: lifted at t_a with D4RT depth, world = camera-t_a (C_{t_a} = I), reprojected through camera C
    held_ok = held[Q[held, t_a]]
    Xh = Xl[held_ok, t_a]
    errs, errs_G, rot_vs_G = [], [], []
    Gi = np.linalg.inv(G)
    for k in range(1, F_pair):
        t = t_a + k
        v = Q[held_ok, t] & ~in_mask(P[held_ok, t], M[t])
        if v.sum() < MIN_PTS or not solved[k]:
            errs.append(np.nan); errs_G.append(np.nan); rot_vs_G.append(np.nan); continue
        errs.append(float(np.nanmedian(np.linalg.norm(reproj_px(Xh[v], E_c[k], K[t]) - P[held_ok[v], t], axis=1))))
        E_d = Gi[t] @ G[t_a]
        errs_G.append(float(np.nanmedian(np.linalg.norm(reproj_px(Xh[v], E_d, K[t]) - P[held_ok[v], t], axis=1))))
        rot_vs_G.append(rot_angle_deg(E_c[k, :3, :3], E_d[:3, :3]))
    errs, errs_G, rot_vs_G = np.array(errs), np.array(errs_G), np.array(rot_vs_G)
    frames = t_a + np.arange(1, F_pair)
    first_win = frames < (bounds[0] if bounds else 10 ** 9)
    med_all = float(np.nanmedian(errs))
    # object path heading under camera C (sanity row): B.5 centroid -> camera-t via G (bookkeeping) -> world via camera C
    with np.load(root / "cache" / "stage_b5" / f"{clip}.npz") as z:
        Xbar = z["X_bar_track"].astype(np.float64)
    kw_ = min(12, F_pair - 1)
    head = None
    if solved[kw_] and np.isfinite(Xbar[[t_a, t_a + kw_]]).all():
        Xw_obj = vipe_camera.object_path_world(Xbar[[t_a, t_a + kw_]], G[[t_a, t_a + kw_]], np.stack([E_c[0], E_c[kw_]]))
        head = angle_deg(Xw_obj[1] - Xw_obj[0], np.asarray(h["d_src"], np.float64))
    # ViPE jerk for comparison (bridged with sigma_ta; jerk is scale-dependent, so both are reported in their own units)
    try:
        with np.load(sorted((root / "bakeoff" / "stage_a5" / clip / "vipe" / "pose").glob("*.npz"))[0]) as z:
            P_c2w = z["data"].astype(np.float64)
        sig = json.loads((root / lock["trajectory"]["sigma_t_record"]).read_text(encoding="utf-8"))[clip]["sigma_abs"][t_a]
        E_v, _ = vipe_camera.phase2_camera(P_c2w, t_a, sig, F_pair, 1e-4)
        jerk_v = jerk(E_v)
    except Exception as e:
        jerk_v = f"n/a ({type(e).__name__})"
    # translation observability (reviewer 2026-10-07): why 035 is held, as a number, for all five clips.
    #   proxy  = median static-point depth at t_a / camera path length over the window (large = translation barely moves the points);
    #   PnP    = translation covariance from the Jacobian at a mid-window frame: eigenvalues of (J^T J)^-1 for the three translation
    #            parameters, in source units per px^2, from the held-out static points (large = poorly constrained).
    z_med = float(np.median(Xl[held_ok, t_a, 2]))
    Cc = np.linalg.inv(E_c[solved])[:, :3, 3]
    path_len = float(np.linalg.norm(np.diff(Cc, axis=0), axis=1).sum()) if len(Cc) > 1 else float("nan")
    kmid = F_pair // 2
    eig = None
    if solved[kmid]:
        tm = t_a + kmid; v = Q[held_ok, tm]
        Xc_ = Xh[v] @ E_c[kmid, :3, :3].T + E_c[kmid, :3, 3]
        fx, fy = K[tm][0, 0], K[tm][1, 1]
        X_, Y_, Z_ = Xc_[:, 0], Xc_[:, 1], Xc_[:, 2]
        Ju = np.stack([fx / Z_, np.zeros_like(Z_), -fx * X_ / Z_ ** 2], 1)          # d(u)/d(t_x, t_y, t_z)
        Jv = np.stack([np.zeros_like(Z_), fy / Z_, -fy * Y_ / Z_ ** 2], 1)
        J = np.concatenate([Ju, Jv], 0)
        try:
            eig = sorted(np.linalg.eigvalsh(np.linalg.inv(J.T @ J)).tolist())
        except np.linalg.LinAlgError:
            eig = None
    observability = {"median_static_depth_over_cam_path": z_med / path_len if path_len > 0 else None, "median_static_depth": z_med, "cam_path_len_C": path_len,
                     "pnp_translation_cov_eigs_mid_window": eig, "worst_axis_sigma_per_px": None if eig is None else float(np.sqrt(max(eig)))}
    rec = {"pair": f"{clip}/{cat}", "t_a": t_a, "F_pair": F_pair, "keyframes": keyframes, "lift_log": log, "translation_observability": observability,
           "n_static_solve": int(len(solve)), "n_static_held": int(len(held_ok)), "frames_solved": int(solved.sum()), "frames_total": int(F_pair),
           "heldout_reproj_px": {"median_all": med_all, "median_first_window": float(np.nanmedian(errs[first_win])) if first_win.any() else None,
                                 "median_after_boundaries": float(np.nanmedian(errs[~first_win])) if (~first_win).any() else None,
                                 "p90_all": float(np.nanquantile(errs, 0.9)), "curve_every4": [None if not np.isfinite(v) else round(float(v), 2) for v in errs[::4]]},
           "G_heldout_reproj_px": {"median_first_window": float(np.nanmedian(errs_G[first_win])) if first_win.any() else None,
                                   "median_after_boundaries": float(np.nanmedian(errs_G[~first_win])) if (~first_win).any() else None},
           "rot_C_vs_G_deg": {"median_first_window": float(np.nanmedian(rot_vs_G[first_win])) if first_win.any() else None, "max": float(np.nanmax(rot_vs_G))},
           "gate": {"patch": lock["trajectory"]["camera_c_gate_patch"], "px_256": gate_px, "outcome": "pass" if med_all <= gate_px and solved.all() else "fail",
                    "reason": None if solved.all() else f"{int((~solved).sum())} frames unsolved"},
           "object_heading_vs_d_src_deg_12f": head, "jerk_centres": {"camera_C": jerk(E_c[solved]) if solved.sum() > 3 else None, "vipe_bridged": jerk_v},
           "instrument": {"pnp": "EPnP+RANSAC, LM refine", "ransac_px": PNP_REPROJ_PX, "split": "track index parity"}}
    out = root / "cache" / "m3" / clip; out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "camera_c.npz", E_c=E_c, solved=solved, t_a=np.int64(t_a), F_pair=np.int64(F_pair), keyframes=np.array(keyframes))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", default=__import__("os").environ.get("MT_ROOT", "/content/drive/MyDrive/motion_transfer"))
    ap.add_argument("--census", action="store_true", help="read cache/census/<clip>/<cat>/handoff.json (source-side, from run_census.py) and write reports/census/")
    ap.add_argument("--clips", nargs="*", default=None)
    args = ap.parse_args(argv)
    lock = H.load_m3_lock()
    assert "camera_c_gate_patch" in lock["trajectory"], "lock m3.trajectory.camera_c_gate_patch missing (reviewer 2026-10-06j: 1 patch); John adds it"
    root = Path(args.drive_root)
    seen, rows = set(), []
    for hp in sorted((root / "cache" / ("census" if args.census else "m2_e")).glob("*/*/handoff.json")):
        clip, cat = hp.parent.parent.name, hp.parent.name
        if clip in seen or (args.clips and clip not in args.clips):
            continue
        seen.add(clip)
        try:
            rows.append(run_clip(root, clip, cat, lock, census=args.census))
        except Exception as e:
            rows.append({"pair": f"{clip}/{cat}", "error": f"{type(e).__name__}: {e}"})
        print(f"[m3 camera C] {clip}: {rows[-1].get('error') or rows[-1]['gate']['outcome']}")
    rep = root / "reports" / ("census" if args.census else "m3_sint"); rep.mkdir(parents=True, exist_ok=True)
    prev = json.loads((rep / "camera_c.json").read_text(encoding="utf-8")) if args.census and (rep / "camera_c.json").exists() else []
    rows_out = [r for r in prev if r["pair"] not in {x["pair"] for x in rows}] + rows if args.census else rows     # census: the record is extended per clip
    (rep / "camera_c.json").write_text(json.dumps(rows_out, indent=1), encoding="utf-8")
    f = lambda v, n=2: "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f"{v:.{n}f}" if isinstance(v, (int, float)) else str(v))   # noqa: E731
    md = ["# Module 3 — camera C: D4RT re-solved from its own static tracks, chained across encoder-window boundaries (held-out gate)", "",
          "| clip | t_a | keyframes | solve / held | frames solved | held-out px: all / first window / after boundaries (p90) | G on the same held-out: first / after | rot C vs G: median first / max | gate | object heading vs d_src @12f | jerk C / ViPE | observability: depth/path, worst-axis σ (src units/px) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            md.append(f"| {r['pair'][:18]} | ERROR {r['error'][:160]} |"); continue
        e, g, rg, gt = r["heldout_reproj_px"], r["G_heldout_reproj_px"], r["rot_C_vs_G_deg"], r["gate"]
        md.append(f"| {r['pair'][:18]} | {r['t_a']} | {r['keyframes']} | {r['n_static_solve']} / {r['n_static_held']} | {r['frames_solved']}/{r['frames_total']} | "
                  f"{f(e['median_all'])} / {f(e['median_first_window'])} / {f(e['median_after_boundaries'])} ({f(e['p90_all'])}) | {f(g['median_first_window'])} / {f(g['median_after_boundaries'])} | "
                  f"{f(rg['median_first_window'])} / {f(rg['max'])} | **{gt['outcome']}** (≤ {f(gt['px_256'])} px){'' if not gt['reason'] else ' ' + gt['reason']} | {f(r['object_heading_vs_d_src_deg_12f'], 1)} | "
                  f"{f(r['jerk_centres']['camera_C'], 4)} / {f(r['jerk_centres']['vipe_bridged'], 4)} | {f(r['translation_observability']['median_static_depth_over_cam_path'], 1)}, {f(r['translation_observability']['worst_axis_sigma_per_px'], 4)} |")
    md += ["", "Reading (reviewer, pre-stated): inside the first window camera C should coincide with G to about the px already measured; the test is whether the "
           "chaining holds the error across the boundaries. A pass admits 035 under `trajectory.source = static_pnp` (fallback, gated); a fail leaves 035 "
           "'camera undetermined'. Jerk is in each camera's own units (camera C in the source gauge, ViPE bridged) and is a shape comparison only."]
    (rep / "camera_c.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
