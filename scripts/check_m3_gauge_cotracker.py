#!/usr/bin/env python
"""The gauge with INDEPENDENT tracks (reviewer 2026-10-07k): the near-field excess seen with D4RT's own 2-D tracks could be tracker lag on
the fastest (nearest) points. CoTracker's tracks (cache/m3/<clip>/cotracker_F<F>_q4.npz, queried at frame 0 on the Wan grid, probe C6
gap 0.31 patch vs Stage-A static tracks) are put through the same time-split triangulation under the bundle-adjusted camera and the
ViPE/sigma_hat camera, with the same band and the same re-binning. Z_lift for a CoTracker point = median lifted depth of D4RT static tracks
within `--nn-px` at t_a whose lifted depths agree within `--nn-rel` (a depth-consistent neighbourhood: at a depth edge a patch-centre
query beside a tree takes the sky's motion and the tree's lift, which is what produced the 3-6x ratios of the first run on 265.2.005).
Also the lift-free comparison Z_tri(CoTracker) / Z_tri(D4RT tracks) at matched positions. The cache starts at frame 0 and holds 81
frames, so any pair whose window ends by frame 80 runs (t_a = 47 pairs included). `--solve`: bundle adjustment on CoTracker's static
tracks from the camera-C-BA initial cameras, held out by parity, as an independent camera candidate (035: if these tracks fit where
D4RT's do not, the tracks were the fault). CPU, Phase 1.

    python scripts/check_m3_gauge_cotracker.py
    python scripts/check_m3_gauge_cotracker.py --clips 000000000035.1.003 --solve
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.check_m3_camera import in_mask, reproj_px                        # noqa: E402
from scripts.refine_m3_camera_ba import ground_band_256, PRIOR_WEIGHT_PX, N_ANCHOR   # noqa: E402
from src.generate.m3 import bundle_adjust as BA, handoff as H                 # noqa: E402
from src.generate.m3.geometry import camera, vipe_camera                       # noqa: E402
from src.preprocess.stage_c import finalize                                   # noqa: E402

WAN_W, WAN_H, SIDE = 832, 480, 256


def run_clip(root: Path, clip: str, cat: str, lock: dict, min_points: int, nn_px: float, dilate_px: int, nn_rel: float = 0.1, solve: bool = False, tracks: str = "cotracker",
             init_px_max: float = 12.0) -> dict:
    import cv2
    h = json.loads((root / "cache" / "m2_e" / clip / cat / "handoff.json").read_text(encoding="utf-8"))
    t_a = int(h["t_a"]); F_pair, _ = camera.frames_for_pair(h["n_frames_after_t_a"])
    pat = {"cotracker": "cotracker_F*_q4.npz", "raft": "raft_F*_*.npz"}[tracks]
    cfs = sorted((root / "cache" / "m3" / clip).glob(pat), key=lambda p: -int(p.stem.split("_F")[1].split("_")[0]))
    assert cfs, f"no {pat} cache for {clip}" + ("" if tracks == "cotracker" else " (run scripts/check_m3_dense_flow.py first)")
    with np.load(cfs[0]) as z:
        tr, vis = z["tracks"].astype(np.float64), z["vis"].astype(bool)      # [F_cache,N,2] at 832x480 from frame 0, [F_cache,N]
    assert tr.shape[2] == 2 and vis.shape == tr.shape[:2]
    assert t_a + F_pair <= tr.shape[0], f"CoTracker cache has {tr.shape[0]} frames; the window needs frames {t_a}..{t_a + F_pair - 1}"
    tr, vis = tr[t_a: t_a + F_pair], vis[t_a: t_a + F_pair]                  # the pair window; k = 0 is t_a
    # Stage A's resize is anisotropic (adapter provenance resize_factors), so the map to the 256^2 grid is linear per axis
    P = np.stack([tr[..., 0] * SIDE / WAN_W, tr[..., 1] * SIDE / WAN_H], -1).transpose(1, 0, 2)   # [N,F,2]
    Q = vis.T & (P[..., 0] >= 0) & (P[..., 0] < SIDE) & (P[..., 1] >= 0) & (P[..., 1] < SIDE)
    tri_px = float(lock["trajectory"]["triangulated_gate_patch"]) * 256.0 / int(lock["lattice"]["Wp"]); gauge_tol = float(lock["trajectory"]["triangulated_gauge_tol"])
    with np.load(root / "cache" / "stage_a" / clip / "bundle.npz") as z:
        K, Xl, Pd, Qd = (z[k].astype(np.float64) if k != "Q" else z[k].astype(bool) for k in ("K", "X_local", "P", "Q"))
    with np.load(root / "cache" / "stage_b" / f"{clip}.npz") as z:
        y_dyn = z["y_dyn"].astype(bool)
    M, _, _ = finalize.read_mask(root / "cache" / "stage_c" / clip)
    win = np.arange(t_a, t_a + F_pair)
    kd = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
    Md = np.stack([cv2.dilate(M[t].astype(np.uint8), kd).astype(bool) for t in win])
    excl = np.stack([in_mask(P[:, k], Md[k]) for k in range(F_pair)], 1)        # [N,F] inside the (dilated) object mask
    static = ~excl.any(1) & (Q.sum(1) >= 8)                                   # CoTracker has no y_dyn: the mask (dilated) is the dynamic exclusion
    # Z_lift proxy: D4RT static tracks within nn_px at t_a whose lifted depths agree within nn_rel (depth-consistent neighbourhood)
    d_static = Qd[:, t_a] & ~y_dyn & ~in_mask(Pd[:, t_a], M[t_a])
    from scipy.spatial import cKDTree
    tree = cKDTree(Pd[d_static, t_a]); zd = Xl[d_static, t_a, 2]
    Z_lift = np.full(len(P), np.nan); n_edge = 0
    for i in np.nonzero(static & Q[:, 0])[0]:
        nb = tree.query_ball_point(P[i, 0], nn_px)
        if len(nb) >= 2:
            z_nb = zd[nb]
            if (z_nb.max() - z_nb.min()) / max(float(np.median(z_nb)), 1e-6) <= nn_rel:
                Z_lift[i] = float(np.median(z_nb))
            else:
                n_edge += 1
    idx = np.nonzero(static)[0]
    # D4RT's own held-out split points under the same cameras, for the lift-free tracker-vs-tracker comparison
    d_idx = np.nonzero(~y_dyn & ~np.stack([Qd[:, t] & in_mask(Pd[:, t], M[t]) for t in range(t_a, t_a + F_pair)], 1).any(1))[0]
    d_idx = d_idx[d_idx % 2 == 1]
    Pd_w, Qd_w = Pd[d_idx][:, t_a: t_a + F_pair], Qd[d_idx][:, t_a: t_a + F_pair]
    excl_d = np.stack([in_mask(Pd[d_idx, t], M[t]) for t in range(t_a, t_a + F_pair)], 1)
    Zl_d = np.where(Qd[d_idx, t_a], Xl[d_idx, t_a, 2], np.nan)
    cb = np.load(root / "cache" / "m3" / clip / "camera_c_ba.npz"); E_ba = cb["E_c"].astype(np.float64)
    cams = {"ba": E_ba}
    pf = sorted((root / "bakeoff" / "stage_a5" / clip / "vipe" / "pose").glob("*.npz")); crec_p = root / lock["trajectory"]["sigma_t_record"]
    if pf and crec_p.exists():
        with np.load(pf[0]) as z:
            P_c2w = z["data"].astype(np.float64)
        sig = float(json.loads(crec_p.read_text(encoding="utf-8"))[clip]["sigma_abs"][t_a])
        cams["vipe_over_sigma"], _ = vipe_camera.phase2_camera(P_c2w, t_a, sig, F_pair, 1e-4)
    band = ground_band_256(root, clip, cat, h, M[t_a])
    out = {"pair": f"{clip}/{cat}", "t_a": t_a, "F_pair": F_pair, "n_cotracker_tracks": int(len(P)), "n_static": int(len(idx)), "n_with_Z_lift": int(np.isfinite(Z_lift[idx]).sum()),
           "n_rejected_depth_edge": int(n_edge), "z_lift_proxy": f"median lifted depth of >=2 D4RT static tracks within {nn_px} px at t_a agreeing within {nn_rel:.0%}",
           "mask_dilate_px": dilate_px, "tracks": tracks, "tracks_file": cfs[0].name, "cameras": {}}
    for name, E in cams.items():
        tri = BA.triangulated_heldout(E, K[win], P[idx], Q[idx], excl[idx], Z_lift[idx], tri_px, gauge_tol, min_points=min_points, return_points=True)
        pts = tri.pop("points")
        tri_d = BA.triangulated_heldout(E, K[win], Pd_w, Qd_w, excl_d, Zl_d, tri_px, gauge_tol, min_points=min_points, return_points=True)
        out["cameras"][name] = {"split": {k: tri[k] for k in ("median_px", "p90_px", "n_points", "dropped", "reproj_ok")},
                                "lift_tercile_information": {k: v for k, v in tri["gauge"].items() if k not in ("outcome", "ok", "read_on", "near")},
                                "band_gauge": BA.band_gauge(pts, band, gauge_tol, min_points=min_points), "rebin": BA.rebin_tables(pts),
                                "cotracker_over_d4rt_Ztri": BA.compare_triangulations(pts, tri_d["points"], radius_px=nn_px, band_mask=band),
                                "d4rt_split_px": tri_d["median_px"]}
    # ΔF rule (reviewer 2026-10-07l, 370): the flow error the tracker choice can cost = band parallax p90 x |β_CoT − β_D4RT| / β_D4RT; ≤ 0.3 patch -> the
    # D4RT-track value is the bridge of record; otherwise the bridge is an interval and the pair carries a low/high sensitivity run
    rj = root / "reports" / "m3_sint" / "camera_c_ba.json"
    d4 = next((x for x in json.loads(rj.read_text(encoding="utf-8")) if x.get("pair", "").startswith(clip)), None) if rj.exists() else None
    beta_d4 = d4["band_gauge"]["ratio"] if d4 and d4.get("band_gauge") else None
    par = d4.get("contact_bridge", {}).get("band_parallax_px") if d4 else None
    beta_cot = out["cameras"]["ba"]["band_gauge"]["ratio"]
    if beta_d4 and beta_cot and par and par.get("p90_px"):
        dF_px = par["p90_px"] * abs(beta_cot - beta_d4) / beta_d4
        dF_patch = dF_px / (256.0 / int(lock["lattice"]["Wp"]))
        out["delta_F"] = {"beta_cotracker": beta_cot, "beta_d4rt": beta_d4, "band_parallax_p90_px": par["p90_px"], "delta_F_px": dF_px, "delta_F_patch": dF_patch,
                          "rule": "<= 0.3 patch: tracker choice immaterial, D4RT value is the bridge; else interval + low/high sensitivity run",
                          "outcome": "immaterial" if dF_patch <= 0.3 else "interval + sensitivity run"}
    else:
        out["delta_F"] = {"beta_cotracker": beta_cot, "beta_d4rt": beta_d4, "note": "needs both band ratios and the D4RT band parallax (rerun refine_m3_camera_ba.py)"}
    if solve:
        out["solve"] = solve_on_cotracker(E_ba, K[win], P, Q, excl, static, Z_lift, band, tri_px, gauge_tol, min_points, init_px_max, root, clip)
    return out


def solve_on_cotracker(E_init, K_w, P, Q, excl, static, Z_lift, band, tri_px, gauge_tol, min_points, init_px_max, root, clip) -> dict:
    """Bundle adjustment on CoTracker's static tracks (parity split: even = solve, odd = held out), initialised from the camera-C-BA
    cameras; points initialised by DLT under the initial cameras. Same instrument settings as refine_m3_camera_ba (prior 200, 40
    anchors nearest the median depth, assembly filter). The held-out split column grades the result; no negative control is run."""
    F = len(E_init)
    idx = np.nonzero(static)[0]; sol_idx, held_idx = idx[idx % 2 == 0], idx[idx % 2 == 1]
    X0, keep = [], []
    for i in sol_idx:
        ks = np.nonzero(Q[i] & ~excl[i])[0]
        if len(ks) < 8:
            continue
        try:
            X0.append(BA.triangulate_dlt(E_init[ks], K_w[ks], P[i, ks])); keep.append(i)
        except ValueError:
            continue
    X0 = np.array(X0); keep = np.array(keep)
    front = X0[:, 2] > 0.05; X0, keep = X0[front], keep[front]
    oc, op, ouv, dropped = [], [], [], {"behind": 0, "outside": 0, "far_from_init": 0}
    for j, i in enumerate(keep):
        for k in range(F):
            if not (Q[i, k] and not excl[i, k]):
                continue
            Xc = X0[j] @ E_init[k, :3, :3].T + E_init[k, :3, 3]
            if Xc[2] <= 0.05:
                dropped["behind"] += 1; continue
            uv0 = reproj_px(X0[j][None], E_init[k], K_w[k])[0]
            if not (0 <= uv0[0] < SIDE and 0 <= uv0[1] < SIDE):
                dropped["outside"] += 1; continue
            if np.linalg.norm(uv0 - P[i, k]) > init_px_max:
                dropped["far_from_init"] += 1; continue
            oc.append(k); op.append(j); ouv.append(P[i, k])
    oc, op, ouv = np.array(oc), np.array(op), np.array(ouv, np.float64)
    used = set(np.unique(op).tolist()); assert len(used) >= 20, "too few CoTracker points survive the assembly filter"
    order = np.argsort(np.abs(X0[:, 2] - np.median(X0[:, 2])))
    anchor = np.array([a for a in order if a in used][:N_ANCHOR])
    E_sol, X_sol, info = BA.bundle_adjust(E_init, X0, oc, op, ouv, K_w, anchor, PRIOR_WEIGHT_PX)
    info["dropped_observations"] = dropped
    tri = BA.triangulated_heldout(E_sol, K_w, P[held_idx], Q[held_idx], excl[held_idx], Z_lift[held_idx], tri_px, gauge_tol, min_points=min_points, return_points=True)
    pts = tri.pop("points")
    tri_init = BA.triangulated_heldout(E_init, K_w, P[held_idx], Q[held_idx], excl[held_idx], Z_lift[held_idx], tri_px, gauge_tol, min_points=min_points)
    from scripts.check_m3_camera import rot_angle_deg
    np.savez_compressed(root / "cache" / "m3" / clip / "camera_cotracker_ba.npz", E_c=E_sol, X_static=X_sol, note="candidate camera from CoTracker static tracks; not consumed by any stage")
    return {"ba": info, "n_solve_points": int(len(keep)), "n_held": int(len(held_idx)), "held_split_px_init": tri_init["median_px"], "held_split_px": tri["median_px"],
            "held_split_p90_px": tri["p90_px"], "n_held_scored": tri["n_points"], "reproj_ok": tri["reproj_ok"],
            "band_gauge": BA.band_gauge(pts, band, gauge_tol, min_points=min_points), "rebin": BA.rebin_tables(pts),
            "rot_change_vs_init_deg_max": float(max(rot_angle_deg(E_sol[k, :3, :3], E_init[k, :3, :3]) for k in range(F))),
            "trans_change_vs_init_rel_max": float(max(np.linalg.norm(E_sol[k, :3, 3] - E_init[k, :3, 3]) / max(np.linalg.norm(E_init[k, :3, 3]), 1e-9) for k in range(1, F)))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", default=__import__("os").environ.get("MT_ROOT", "/content/drive/MyDrive/motion_transfer"))
    ap.add_argument("--clips", nargs="+", default=None)
    ap.add_argument("--nn-px", type=float, default=4.0)
    ap.add_argument("--dilate-px", type=int, default=6)
    ap.add_argument("--nn-rel", type=float, default=0.10, help="lifted depths of the matched D4RT neighbours must agree within this (depth-consistent proxy)")
    ap.add_argument("--solve", action="store_true", help="also bundle-adjust on CoTracker's static tracks (independent camera candidate, parity held-out)")
    ap.add_argument("--tracks", default="cotracker", choices=["cotracker", "raft"], help="the 2-D tracks: CoTracker's q4 cache or RAFT's (scripts/check_m3_dense_flow.py); "
                    "raft also re-measures the carried beta against the pixel bridge (reviewer 2026-10-11 round 4)")
    ap.add_argument("--force", action="store_true", help="replace rows already recorded for the same pair (rule 24: refused by default)")
    args = ap.parse_args(argv)
    lock = H.load_m3_lock(); root = Path(args.drive_root)
    min_points = int(__import__("yaml").safe_load(Path("configs/m3.yaml").read_text(encoding="utf-8"))["m3"]["trajectory"]["triangulated_min_points"])
    rows, seen = [], set()
    for hp in sorted((root / "cache" / "m2_e").glob("*/*/handoff.json")):
        clip, cat = hp.parent.parent.name, hp.parent.name
        if clip in seen or (args.clips and clip not in args.clips) or not (root / "cache" / "m3" / clip / "camera_c_ba.npz").exists():
            continue
        seen.add(clip)
        try:
            r = run_clip(root, clip, cat, lock, min_points, args.nn_px, args.dilate_px, nn_rel=args.nn_rel, solve=args.solve, tracks=args.tracks)
            if args.tracks == "raft" and "error" not in r:
                # the carried beta re-measured: rho_band under each camera x Z_L_gnd_base / Z_M_c of the bridge record; registered against the pixel bridge's overall beta_pix
                # rho_band is a clip property; the carried beta and its pixel-bridge reading are per PAIR: every pair of the clip with a bridge row is re-measured
                bi = root / "reports" / "m3_sint" / "bridge_interval.json"
                brows = [x for x in (json.loads(bi.read_text(encoding="utf-8")) if bi.exists() else []) if x.get("pair", "").startswith(clip + "/") and x.get("camera") == "vipe_over_sigma"]
                r["beta_remeasured"] = {}
                for brow in brows:
                    pcat = brow["pair"].split("/")[1]
                    pbj = next((root / "reports" / "m3_s4" / f"pixel_bridge_{clip}_{pcat}{sfx}.json" for sfx in ("", "_raft", "_d4rt") if (root / "reports" / "m3_s4" / f"pixel_bridge_{clip}_{pcat}{sfx}.json").exists()), None)
                    bpix = json.loads(pbj.read_text(encoding="utf-8"))["overall"].get("beta_pix") if pbj else None
                    if pbj is not None and pbj.name.endswith(("_raft.json", "_d4rt.json")):
                        print(f"[gauge raft] beta_pix read from {pbj.name} (the CoTracker pixel-bridge file is absent on this root)", flush=True)
                    for name, c in r["cameras"].items():
                        rho_r = c["band_gauge"].get("ratio"); det = c["band_gauge"].get("outcome") == "determined"
                        beta_r = (rho_r * brow["Z_L_gnd_base"] / brow["Z_M_c"]) if (rho_r is not None and brow.get("Z_L_gnd_base") and brow.get("Z_M_c")) else None
                        rel = abs(beta_r / bpix - 1.0) if (beta_r and bpix) else None
                        r["beta_remeasured"][f"{pcat}|{name}"] = {"pair": brow["pair"], "camera": name, "rho_band_raft": rho_r, "determined": det, "n_band_points": c["band_gauge"].get("n_band_points"),
                                                                 "beta_carried_raft": beta_r, "beta_pix_overall": bpix, "rel_diff": rel, "registered_within_5pct": (rel <= 0.05) if rel is not None else None,
                                                                 "beta_of_record": (brow.get("record") or {}).get("beta_rec")}
                        print(f"[gauge raft] {brow['pair']} camera {name}: rho_band {rho_r} ({'determined' if det else 'not determined'}, {c['band_gauge'].get('n_band_points')} pts) -> carried beta {beta_r} "
                              f"vs pixel-bridge beta_pix {bpix}: {'held' if (rel is not None and rel <= 0.05) else ('FAILED' if rel is not None else 'undefined')}", flush=True)
        except Exception as e:
            r = {"pair": f"{clip}/{cat}", "tracks": args.tracks, "error": f"{type(e).__name__}: {e}"}   # the error row keeps its tracks key (rule 24 keys on it)
            print(f"[gauge {args.tracks}] {clip}: ERROR {r['error']}", flush=True)
        rows.append(r); print(f"[gauge cotracker] {clip}: {r.get('error') or {k: v['band_gauge']['ratio'] for k, v in r['cameras'].items()}}")
    rep = root / "reports" / "m3_sint"; rep.mkdir(parents=True, exist_ok=True)
    jf = rep / "gauge_cotracker.json"
    prev = json.loads(jf.read_text(encoding="utf-8")) if jf.exists() else []                 # rule 24: a run extends the archived record
    key = lambda r: (r["pair"], "solve" in r, r.get("tracks", "cotracker"))                   # noqa: E731
    dup = {key(r) for r in prev} & {key(r) for r in rows}
    assert not dup or args.force, f"rule 24: {len(dup)} rows already recorded in {jf.name} ({sorted(dup)}); pass --force to replace them"
    new_keys = {key(r) for r in rows}
    rows = sorted([r for r in prev if key(r) not in new_keys] + rows, key=lambda r: (r["pair"], "solve" in r, r.get("tracks", "cotracker")))
    jf.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    f = lambda v, n=3: "-" if v is None else f"{v:.{n}f}"                                   # noqa: E731
    md = ["# Module 3 — the gauge with CoTracker's tracks (independent of D4RT's 2-D tracks; reviewer 2026-10-07k)", "",
          "| clip | camera | tracks static / with Z_lift (rejected at depth edges) | split px CoTracker (n) / D4RT | Z_tri CoTracker ÷ Z_tri D4RT at matched points: median [CI] (n); by row top/middle/bottom | BAND ratio [CI] (n; fit → verify) → outcome | by lift depth near/mid/far | by triangulated depth near/mid/far | by image row top/middle/bottom |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            md.append(f"| {r['pair'][:18]} | ERROR {r['error'][:160]} |"); continue
        for name, c in r["cameras"].items():
            bg, rb, s = c["band_gauge"], c["rebin"], c["split"]
            tb = lambda key: "/".join(f(v["median"]) for v in rb.get(key, {}).values()) or "-"                                        # noqa: E731
            cmp_ = c["cotracker_over_d4rt_Ztri"]
            cs = ("-" if cmp_["n_matched"] < 5 else f"{f(cmp_['median'])} [{f(cmp_['ci95'][0])}, {f(cmp_['ci95'][1])}] (n={cmp_['n_matched']}); "
                  + "/".join(f(v["median"]) for v in cmp_.get("by_image_row", {}).values()))
            md.append(f"| {r['pair'][:18]} | {name} | {r['n_static']} / {r['n_with_Z_lift']} ({r['n_rejected_depth_edge']}) | {f(s['median_px'], 2)} ({s['n_points']}) / {f(c['d4rt_split_px'], 2)} | {cs} | "
                      f"{f(bg['ratio'])} [{'-' if bg['ci95'] is None else f(bg['ci95'][0]) + ', ' + f(bg['ci95'][1])}] (n={bg['n_band_points']}; {f(bg['half_fit'])} → {f(bg['half_verify'])}) → {bg['outcome']} | "
                      f"{tb('by_lift_depth')} | {tb('by_triangulated_depth')} | {tb('by_image_row')} |")
    md += ["", "| clip | β CoTracker | β D4RT | band parallax p90 px | ΔF px / patch | rule outcome |", "|---|---|---|---|---|---|"]
    for r in rows:
        dF = r.get("delta_F") if "error" not in r else None
        if dF:
            md.append(f"| {r['pair'][:18]} | {f(dF.get('beta_cotracker'))} | {f(dF.get('beta_d4rt'))} | {f(dF.get('band_parallax_p90_px'), 1)} | "
                      f"{f(dF.get('delta_F_px'), 2)} / {f(dF.get('delta_F_patch'), 3)} | {dF.get('outcome') or dF.get('note')} |")
    for r in rows:
        if "solve" in r:
            sv = r["solve"]; bg = sv["band_gauge"]
            md += ["", f"**{r['pair'][:18]}, bundle adjustment on CoTracker's static tracks** (solve {sv['n_solve_points']} points / {sv['ba']['n_obs']} obs, held out {sv['n_held']}): "
                   f"reproj RMS {sv['ba']['reproj_rms_before']:.2f} → {sv['ba']['reproj_rms_after']:.2f} px; held-out split {f(sv['held_split_px_init'], 2)} (initial cameras) → "
                   f"**{f(sv['held_split_px'], 2)} px** (p90 {f(sv['held_split_p90_px'], 2)}; n={sv['n_held_scored']}; gate {'pass' if sv['reproj_ok'] else 'fail'}); "
                   f"rotation change vs camera-C-BA {sv['rot_change_vs_init_deg_max']:.1f}°, translation change {sv['trans_change_vs_init_rel_max']:.0%} max; "
                   f"band ratio {f(bg['ratio'])} (n={bg['n_band_points']}) → {bg['outcome']}. Camera saved as cache/m3/<clip>/camera_cotracker_ba.npz (candidate, consumed by nothing)."]
    md += ["", "Reading fixed in advance: the same near excess with CoTracker's tracks as with D4RT's -> the lift is the cause (tracks cleared); a smaller or absent "
           "excess -> D4RT's tracks lag on the nearest points and the D4RT-track ratio overstates the bias. The band ratio from CoTracker is the one to set beside the "
           "D4RT-track band ratio before any bridge amendment; Z_lift here is a nearest-neighbour proxy from D4RT's static tracks (stated above)."]
    (rep / "gauge_cotracker.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
