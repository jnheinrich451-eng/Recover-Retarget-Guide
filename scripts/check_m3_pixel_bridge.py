#!/usr/bin/env python
"""Reviewer 2026-10-11, CPU checks 2 and 3 before any choice of record for the bridge.

Check 2, the pixel bridge decomposed: on covered static patches (the render mask at frame 0), the bridge beta' that best predicts the SOURCE
clip's own 2-D motion (CoTracker on the source, the S2 tracks) from the target geometry, re-rendering the flow at each beta' (the camera
translation scales by beta_rec / beta', the rotation does not):  beta_pix = argmin_beta' sum_{q,t} || u_src_q(t) - F_q(t; beta') ||.
Reported overall, by target-depth tercile and on the contact strip. Constant across bins = a global scale (D-M3-08 amendment 2 candidate);
beta_pix ~ beta_rec at the contact and larger away from it = the target's depth shape (§2.6), not beta.

Check 3, depth at the same pixels over the covered region: MoGe x k_bg (the target's depth, the lift the render uses) against D4RT's lift at
t_a and against the triangulated depth of the camera-C bundle-adjustment held-out points (both in the source's gauge; the band ratio rho
relates lift to triangulation).

CPU, read-only on the caches; writes reports/m3_s4/pixel_bridge_<clip>_<cat>.{json,md}. Default paths are the Drive layout; --cache-m3 /
--reports let it run on the repo's local copies.

    python scripts/check_m3_pixel_bridge.py --pair 000000000265.2.005/vehicle --frames 33
    python scripts/check_m3_pixel_bridge.py --pair 000000000265.2.005/vehicle --frames 33 --cache-m3 measurements/m3 --reports measurements
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.generate.m3 import render_target as RT                                   # noqa: E402

WAN_W, WAN_H, PX = 832, 480, 16
HELDOUT_GRID = 256.0                                                               # the BA held-out points carry uv at 256x256


def project(K, E, X):
    """X [N,3] world -> pixels [T,N,2] and depth [T,N] under E [T,4,4]."""
    Xc = np.einsum("tij,nj->tni", E[:, :3, :3], X) + E[:, None, :3, 3]
    p = np.einsum("ij,tnj->tni", K, Xc)
    return p[..., :2] / np.maximum(p[..., 2:3], 1e-9), Xc[..., 2]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", default=__import__("os").environ.get("MT_ROOT", "/content/drive/MyDrive/motion_transfer"))
    ap.add_argument("--cache-m3", default=None, help="override: the m3 cache folder (local copies: measurements/m3)")
    ap.add_argument("--reports", default=None, help="override: the reports folder (local copies: measurements)")
    ap.add_argument("--pair", required=True)
    ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--bridge", default="rec")
    ap.add_argument("--nn-px", type=float, default=8.0, help="check 3: match a query to a held-out point within this many Wan pixels")
    ap.add_argument("--min-pts", type=int, default=3)
    ap.add_argument("--min-vis", type=int, default=4, help="a query enters with at least this many visible latent frames after frame 0 (of frames//4)")
    ap.add_argument("--strip-px", type=int, default=64, help="contact strip: covered pixels within this distance of the object's bottom tenth")
    ap.add_argument("--tracks", default="cotracker", choices=["cotracker", "d4rt", "raft"], help="the source's 2-D motion: CoTracker (S2 cache) or D4RT's own tracks "
                    "(Stage A bundle P/Q, an independent tracker; reviewer 2026-10-11 §4 arm 2)")
    ap.add_argument("--lift", default="lift.npz", help="the lift file to take the target geometry from (lift_aligned.npz after scripts/align_m3_lift.py: the background is then in "
                    "the camera's own gauge and beta_pix is expected at 1.0 in every tercile)")
    ap.add_argument("--sigma-record", default=None, help="the A.5 calibration record with sigma_abs per frame (default <drive-root>/bakeoff/stage_a5/calibration_record.json, "
                    "or <reports>/stage_a5/calibration_record.json with --reports)")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import cv2
    from scipy.spatial import cKDTree
    root = Path(args.drive_root); clip, cat = args.pair.split("/")
    cm3 = Path(args.cache_m3) if args.cache_m3 else root / "cache" / "m3"
    rep = Path(args.reports) if args.reports else root / "reports"
    d = cm3 / clip / cat
    tag = "" if args.bridge == "rec" else f"_beta_{args.bridge}"
    with np.load(d / f"geometry{tag}.npz") as z:
        E_full, beta_rec, src_frames = z["E_full"].astype(np.float64), float(z["beta"]), z["src_frames"].astype(int)
    with np.load(d / args.lift) as z:
        Dsq, msq, amask_sq, Kq, K_wan = z["depth_src_sq"].astype(np.float64), z["mask_sq"].astype(bool), z["anchor_mask_sq"].astype(bool), z["K_sq"].astype(np.float64), z["K_wan"].astype(np.float64)
        X_bg_sparse, sky_l = z["X_bg"].astype(np.float64), z["sky_lattice"].astype(bool)
    lift_tag = "" if args.lift == "lift.npz" else "_" + Path(args.lift).stem
    with np.load(d / "render.npz") as z:
        region = z["mask_px"][0].astype(bool)
    assert np.allclose(E_full[0], np.eye(4), atol=1e-9)
    t_a = int(src_frames[0]); T_v = 1 + (args.frames - 1) // 4; frames_l = np.arange(T_v) * 4
    # the same dense lift the render uses (background only), plus the object's own pixels for the contact strip
    dummy = np.zeros(Dsq.shape + (3,), np.uint8)
    X, _ = RT.lift_dense(Dsq, msq & ~amask_sq, Kq, dummy)
    Xo, _ = RT.lift_dense(Dsq, msq & amask_sq, Kq, dummy)
    p0, z0 = project(K_wan, E_full[:1], X); p0, z0 = p0[0], z0[0]
    inside0 = (z0 > 0) & (p0[:, 0] >= 0) & (p0[:, 0] < WAN_W) & (p0[:, 1] >= 0) & (p0[:, 1] < WAN_H)
    X, p0, z0 = X[inside0], p0[inside0], z0[inside0]
    cell = (p0[:, 1] // PX).astype(int) * (WAN_W // PX) + (p0[:, 0] // PX).astype(int)
    # the contact strip: covered pixels within --strip-px of the bottom tenth of the object's frame-0 footprint (the render mask itself keeps
    # 16 px clear of the object: footprint dilation 8 + erosion 8, so a 16 px strip holds no query on this pair)
    po, zo = project(K_wan, E_full[:1], Xo); po, zo = po[0], zo[0]
    fp = np.zeros((WAN_H, WAN_W), bool); oki = (zo > 0) & (po[:, 0] >= 0) & (po[:, 0] < WAN_W) & (po[:, 1] >= 0) & (po[:, 1] < WAN_H)
    fp[po[oki, 1].astype(int), po[oki, 0].astype(int)] = True
    rows_o = np.nonzero(fp.any(1))[0]; strip = np.zeros_like(fp)
    if len(rows_o):
        lo = int(rows_o.max() - 0.10 * (rows_o.max() - rows_o.min() + 1)); strip[lo: rows_o.max() + 1] = fp[lo: rows_o.max() + 1]
        k_s = 2 * args.strip_px + 1
        strip = cv2.dilate(strip.astype(np.uint8), np.ones((k_s, k_s), np.uint8)).astype(bool) & ~cv2.dilate(fp.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    # the source's own 2-D motion: CoTracker on the source clip (frame-0 queries on the Wan grid), the clip's window from t_a
    if args.tracks in ("cotracker", "raft"):
        pat = "cotracker_F*_q4.npz" if args.tracks == "cotracker" else "raft_F*_*.npz"        # raft: scripts/check_m3_dense_flow.py (q4 queries or its own grid), the same format
        cands = sorted((cm3 / clip).glob(pat), key=lambda p: -int(p.stem.split("_F")[1].split("_")[0]))
        assert cands, f"no {pat} in {cm3 / clip}" + ("" if args.tracks == "cotracker" else ": run scripts/check_m3_dense_flow.py first")
        cf = cands[0]
        with np.load(cf) as z:
            tr, vis, qf = z["tracks"].astype(np.float64), z["vis"].astype(np.float64) > 0.5, int(z["query_frame"])
        assert qf == t_a, (qf, t_a)
        assert tr.shape[0] >= args.frames, (tr.shape, args.frames)
        tr, vis = tr[:args.frames], vis[:args.frames]
    else:
        # D4RT's own tracks: Stage A bundle P [N,F,2] on the 256^2 grid (anisotropic resize of the Wan frame), Q [N,F] visibility; the window from t_a
        cf = cm3.parent / "stage_a" / clip / "bundle.npz"
        assert cf.exists(), f"{cf} missing: the D4RT arm needs the Stage A bundle (Phase 1 on Colab)"
        with np.load(cf) as z:
            Pd, Qd = z["P"].astype(np.float64), z["Q"].astype(bool)
        assert Pd.shape[1] >= t_a + args.frames, (Pd.shape, t_a, args.frames)
        tr = np.stack([Pd[:, t_a: t_a + args.frames, 0] * WAN_W / 256.0, Pd[:, t_a: t_a + args.frames, 1] * WAN_H / 256.0], -1).transpose(1, 0, 2)   # [F,N,2] Wan px
        vis = Qd[:, t_a: t_a + args.frames].T
        yd = cm3.parent / "stage_b" / f"{clip}.npz"
        if yd.exists():
            with np.load(yd) as z:
                vis = vis & ~z["y_dyn"].astype(bool)[None]                          # Stage B's dynamic tracks are not static background
        vis = vis & np.isfinite(tr).all(-1)
        tr = np.where(np.isfinite(tr), tr, 0.0)
        tr[0] = np.where(vis[0][:, None], tr[0], -1e6)                             # a track not visible at frame 0 has no query: pushed off the frame
    lift_tag += {"cotracker": "", "d4rt": "_d4rt", "raft": "_raft"}[args.tracks]
    q0 = tr[0]; qi, qj = np.clip(q0[:, 1].astype(int), 0, WAN_H - 1), np.clip(q0[:, 0].astype(int), 0, WAN_W - 1)
    qcell = (qi // PX) * (WAN_W // PX) + (qj // PX)
    vis_l = vis[frames_l]                                                              # [T_v, N]
    covered = region[qi, qj] & (vis_l[1:].sum(0) >= args.min_vis)
    # points per query cell (frame-0 projection); the target's depth at the query = the median depth of those points
    order = np.argsort(cell, kind="stable"); cs = cell[order]
    starts = np.searchsorted(cs, np.arange(cs.max() + 2))
    members = {int(c): order[starts[c]: starts[c + 1]] for c in np.unique(cs)}
    pts_q = [members.get(int(c), np.zeros(0, int)) for c in qcell]
    npts = np.array([len(p) for p in pts_q])
    sel = covered & (npts >= args.min_pts)
    z_tgt = np.array([float(np.median(z0[p])) if len(p) else np.nan for p in pts_q])
    on_strip = strip[qi, qj]
    idx = np.nonzero(sel)[0]
    print(f"[pixel bridge] {args.pair}: {len(idx)} covered static queries with >= {args.min_pts} lifted points ({int(on_strip[idx].sum())} on the contact strip); "
          f"beta_rec {beta_rec:.3f}; {len(X)} lifted background points; tracks {cf.name}")
    u_src = (tr[frames_l] - tr[0][None]) / PX                                        # [T_v, N, 2] patches
    betas = np.exp(np.linspace(np.log(0.6), np.log(4.0), 77))

    def flow_at(bp):
        E = E_full[frames_l].copy(); E[:, :3, 3] *= beta_rec / bp
        p, zz = project(K_wan, E, X)
        return (p - p0[None]) / PX, zz > 0

    kr = int(np.argmin(np.abs(betas - beta_rec)))
    E_grid = np.full((len(betas), len(idx)), np.nan); mag_src = np.full(len(idx), np.nan); mag_tgt = np.full(len(idx), np.nan)
    E_j = np.full((len(betas), T_v - 1, len(idx)), np.nan)                           # per-frame errors, for beta_pix(j)
    for b, bp in enumerate(betas):
        Fb, front = flow_at(bp); fa = front.all(0)
        for k, q in enumerate(idx):
            pts = pts_q[q]; pts = pts[fa[pts]]
            if len(pts) < args.min_pts:
                continue
            Fq = np.median(Fb[:, pts], axis=1)                                        # [T_v, 2], the patch's rendered flow
            vq = vis_l[1:, q]
            e_ = np.linalg.norm(u_src[1:, q] - Fq[1:], axis=1)
            E_grid[b, k] = e_[vq].mean(); E_j[b, vq, k] = e_[vq]
            if b == kr:
                mag_src[k] = np.linalg.norm(u_src[1:, q], axis=1)[vq].mean(); mag_tgt[k] = np.linalg.norm(Fq[1:], axis=1)[vq].mean()
    valid = np.isfinite(E_grid).all(0)
    zq = z_tgt[idx]; bounds = np.nanquantile(zq[valid], [1 / 3, 2 / 3]); tq = np.digitize(zq, bounds)

    def fit(mask):
        n = int(mask.sum())
        if n < 10:
            return {"n": n, "beta_pix": None}
        tot = E_grid[:, mask].mean(1); k = int(np.argmin(tot))
        own = betas[np.argmin(E_grid[:, mask], axis=0)]                              # each patch's own best beta'
        return {"n": n, "beta_pix": float(betas[k]), "beta_pix_over_beta_rec": float(betas[k] / beta_rec), "residual_at_beta_pix_patch": float(tot[k]),
                "mean_disp_src_patch": float(np.median(mag_src[mask])), "mean_disp_target_at_beta_rec_patch": float(np.median(mag_tgt[mask])),
                "target_over_src_at_same_pixels": float(np.median(mag_tgt[mask] / np.maximum(mag_src[mask], 1e-6))),
                "residual_at_beta_rec_patch": float(tot[kr]), "per_query_beta_quartiles": [float(v) for v in np.quantile(own, [0.25, 0.5, 0.75])],
                "at_grid_edge": bool(k in (0, len(betas) - 1)), "curve": [[round(float(b), 3), round(float(v), 3)] for b, v in zip(betas[::4], tot[::4])]}
    # beta_pix per latent frame over the whole covered set: a ratio that grows with time is the camera's translation magnitude over time, not depth
    per_frame = []
    for j in range(1, T_v):
        tot = np.nanmean(E_j[:, j - 1, valid], axis=1); k = int(np.argmin(tot))
        per_frame.append({"j": j, "n": int(np.isfinite(E_j[0, j - 1, valid]).sum()), "beta_pix": float(betas[k]), "beta_pix_over_beta_rec": float(betas[k] / beta_rec),
                          "residual_at_beta_pix": float(tot[k]), "residual_at_beta_rec": float(tot[kr])})
    # ---- reviewer 2026-10-11 §4, arm 1: per frame, refit a small rotation correction dR_t (3 params) + translation scale s_t, against scale alone
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as Rot
    Kinv = np.linalg.inv(K_wan)
    rep_ok = valid & np.isfinite(zq)
    Xrep = (Kinv @ np.stack([q0[idx, 0], q0[idx, 1], np.ones(len(idx))], 0)).T * zq[:, None]     # one point per query at its cell's target depth (camera 0 = world)

    def flow_rep(Rm, tv, sel):
        Xc = Xrep[sel] @ Rm.T + tv
        p = (K_wan @ Xc.T).T; p = p[:, :2] / np.maximum(p[:, 2:3], 1e-9)
        return (p - q0[idx][sel]) / PX
    rot_rows = []
    for j in range(1, T_v):
        Rj, tj = E_full[frames_l[j], :3, :3], E_full[frames_l[j], :3, 3]
        sel = rep_ok & vis_l[j, idx]
        if sel.sum() < 30:
            rot_rows.append({"j": j, "n": int(sel.sum())}); continue
        tgt = u_src[j, idx][sel]

        def res_s(p):
            return (flow_rep(Rj, p[0] * tj, sel) - tgt).ravel()

        def res_rs(p):
            return (flow_rep(Rot.from_rotvec(p[:3]).as_matrix() @ Rj, p[3] * tj, sel) - tgt).ravel()
        f_s = least_squares(res_s, [1.0], loss="huber", f_scale=0.5)
        f_rs = least_squares(res_rs, [0.0, 0.0, 0.0, f_s.x[0]], loss="huber", f_scale=0.5)
        rot_rows.append({"j": j, "n": int(sel.sum()), "s_scale_only": float(f_s.x[0]), "s_with_rotation": float(f_rs.x[3]),
                         "dtheta_deg": float(np.degrees(np.linalg.norm(f_rs.x[:3]))), "drotvec_deg": [float(np.degrees(v)) for v in f_rs.x[:3]],
                         "resid_scale_only": float(np.median(np.linalg.norm(res_s(f_s.x).reshape(-1, 2), axis=1))),
                         "resid_with_rotation": float(np.median(np.linalg.norm(res_rs(f_rs.x).reshape(-1, 2), axis=1))),
                         "s_by_tercile_after_rotation": {n: (float(least_squares(lambda p, m=(sel & (tq == k_)): (flow_rep(Rot.from_rotvec(f_rs.x[:3]).as_matrix() @ Rj, p[0] * tj, m) - u_src[j, idx][m]).ravel(), [1.0], loss="huber", f_scale=0.5).x[0])
                                                            if (sel & (tq == k_)).sum() >= 10 else None) for k_, n in enumerate(("near", "mid", "far"))}})
    # the ViPE-vs-D4RT rotation gap on this clip (S-int camera check, curve every 8 frames), interpolated to the latent frames
    rot_gap = None
    cc = rep / "m3_sint" / "camera_check.json"
    if cc.exists():
        rows_cc = [r for r in json.loads(cc.read_text(encoding="utf-8")) if r.get("pair", "").startswith(clip)]
        if rows_cc and isinstance(rows_cc[-1].get("rot_d4rt_vs_vipe_deg"), dict) and rows_cc[-1]["rot_d4rt_vs_vipe_deg"].get("curve_every8"):
            cur = np.asarray(rows_cc[-1]["rot_d4rt_vs_vipe_deg"]["curve_every8"], np.float64)
            rot_gap = [float(np.interp(t_a + f_, np.arange(len(cur)) * 8, cur)) for f_ in frames_l[1:]]
    frame_tercile = {}
    for k_, name in enumerate(("near", "mid", "far")):
        row = []
        for j in range(1, T_v):
            msk = valid & (tq == k_)
            tot = np.nanmean(E_j[:, j - 1, msk], axis=1) if msk.sum() >= 10 else None
            row.append(None if tot is None or not np.isfinite(tot).any() else float(betas[int(np.nanargmin(tot))] / beta_rec))
        frame_tercile[name] = row
    sig_path = Path(args.sigma_record) if args.sigma_record else ((rep / "stage_a5" / "calibration_record.json") if args.reports else (root / "bakeoff" / "stage_a5" / "calibration_record.json"))
    sigma_ratio = None
    if sig_path.exists():
        srec = json.loads(sig_path.read_text(encoding="utf-8")).get(clip, {})
        sa = srec.get("sigma_abs")
        if sa is not None and len(sa) > t_a + frames_l[-1]:
            sa = np.asarray(sa, np.float64)
            sigma_ratio = [float(sa[t_a + f_] / sa[t_a]) for f_ in frames_l[1:]]
    out = {"pair": args.pair, "frames": args.frames, "tracks": args.tracks, "lift": args.lift, "beta_pix_per_frame": per_frame, "beta_pix_over_beta_rec_frame_x_tercile": frame_tercile,
           "rotation_refit_per_frame": rot_rows, "rot_d4rt_vs_vipe_deg_interp": rot_gap,
           "sigma_hat_t_over_sigma_hat_ta": sigma_ratio, "sigma_record": str(sig_path) if sig_path.exists() else None, "bridge_of_geometry": args.bridge, "beta_rec": beta_rec, "t_a": t_a, "n_queries": int(valid.sum()), "tracks": cf.name,
           "beta_grid": [float(betas[0]), float(betas[-1]), len(betas)], "overall": fit(valid),
           "by_target_depth_tercile": {n: fit(valid & (tq == k)) for k, n in enumerate(("near", "mid", "far"))}, "contact_strip": fit(valid & on_strip[idx]),
           "target_depth_tercile_bounds": [float(v) for v in bounds]}
    print("  beta_pix overall", out["overall"].get("beta_pix"), "| terciles", {k: v.get("beta_pix") for k, v in out["by_target_depth_tercile"].items()}, "| strip", out["contact_strip"].get("beta_pix"))
    # ---- check 3: depth at the same pixels over the covered region
    dep = {}
    bi = rep / "m3_sint" / "bridge_interval.json"
    rho = None
    if bi.exists():
        rows = [r for r in json.loads(bi.read_text(encoding="utf-8")) if r.get("pair") == args.pair and r.get("camera") == "vipe_over_sigma"]
        rho = rows[-1].get("rho_band_median") if rows else None
    dep["rho_band_median_vipe"] = rho
    hp_ = cm3 / clip / "camera_c_ba_heldout.npz"
    if hp_.exists():
        with np.load(hp_, allow_pickle=True) as z:
            pts = [p for p in json.loads(str(z["points"])) if p.get("uv_ta") is not None and p.get("Z_lift") and p.get("Z_tri")]
            dep["band_gauge"] = json.loads(str(z["band_gauge"]))
        uv = np.array([p["uv_ta"] for p in pts], np.float64) * np.array([WAN_W / HELDOUT_GRID, WAN_H / HELDOUT_GRID])
        ztri = np.array([p["Z_tri"] for p in pts]); zlift = np.array([p["Z_lift"] for p in pts])
        dist, j = cKDTree(uv).query(q0[idx]); ok = (dist <= args.nn_px) & valid
        r_tri, r_lift, tq_ok = zq[ok] / ztri[j[ok]], zq[ok] / zlift[j[ok]], tq[ok]
        med = lambda a: float(np.median(a)) if len(a) >= 5 else None                 # noqa: E731
        dep.update({"n_matched": int(ok.sum()), "n_heldout_points": len(pts), "moge_kbg_over_Ztri_median": med(r_tri), "moge_kbg_over_lift_median": med(r_lift),
                    "lift_over_Ztri_median_at_matched": med(zlift[j[ok]] / ztri[j[ok]]),
                    "by_target_depth_tercile": {n: {"n": int((tq_ok == k).sum()), "over_Ztri": med(r_tri[tq_ok == k]), "over_lift": med(r_lift[tq_ok == k])} for k, n in enumerate(("near", "mid", "far"))},
                    "note": "Z_tri: camera-C bundle adjustment (prior 200), the source's gauge; lift: D4RT at t_a, the source's gauge; MoGe x k_bg: the target's depth as the "
                            "render lifts it (camera t_a, target gauge). The bridge beta divides the camera's translations, so a constant MoGe/Z_tri ratio across depth is the "
                            "bridge's premise and a ratio that trends with depth is the target's depth shape."})
        print("  depth: MoGe*k_bg / Z_tri", dep["moge_kbg_over_Ztri_median"], "| / lift", dep["moge_kbg_over_lift_median"], "| lift/Z_tri", dep["lift_over_Ztri_median_at_matched"],
              "| terciles", {k: (v["over_Ztri"], v["over_lift"], v["n"]) for k, v in dep["by_target_depth_tercile"].items()})
    # ---- the grader's "expected" column re-taken at the grader's own sampling (rule 15: the instrument first). The grader's expected flow is the
    # median over the SPARSE X_bg points inside the region; its measured value is the median over a uniform pixel grid of tracks visible at frame j.
    # Those are different point sets; here both are taken at the same uniform queries (region, not object, not sky, visible at j).
    sky_wan = cv2.resize(sky_l.astype(np.uint8), (WAN_W, WAN_H), interpolation=cv2.INTER_NEAREST).astype(bool)
    fp_d = cv2.dilate(fp.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
    bgq = region[qi, qj] & ~fp_d[qi, qj] & ~sky_wan[qi, qj]
    F_q, _ = RT.flow_at_queries(X, E_full[frames_l], K_wan, q0, WAN_H, WAN_W, cell_px=PX, min_pts=args.min_pts); F_q = F_q / PX
    pbs, zbs = project(K_wan, E_full[frames_l], X_bg_sparse)
    ok0 = (zbs[0] > 0) & (pbs[0, :, 0] >= 0) & (pbs[0, :, 0] < WAN_W) & (pbs[0, :, 1] >= 0) & (pbs[0, :, 1] < WAN_H)
    pin = region[np.clip(pbs[0, :, 1].astype(int), 0, WAN_H - 1), np.clip(pbs[0, :, 0].astype(int), 0, WAN_W - 1)]
    per_j = []
    for j in range(T_v):
        vq = bgq & vis_l[j] & np.isfinite(F_q[j]).all(1)
        per_j.append({"j": j, "n": int(vq.sum()), "source_median_patch": float(np.median(np.linalg.norm(u_src[j][vq], axis=1))) if vq.sum() >= 5 else None,
                      "target_at_same_queries_median_patch": float(np.median(np.linalg.norm(F_q[j][vq], axis=1))) if vq.sum() >= 5 else None,
                      "grader_expected_sparse_Xbg_patch": float(np.median(np.linalg.norm((pbs[j] - pbs[0])[ok0 & pin & (zbs[j] > 0)], axis=1) / PX))})
    out["grader_expected_retaken"] = {"note": "per latent frame j: the source's own tracked displacement (CoTracker, grader-equivalent selection: render mask, not object (17 px dilated), "
                                              "not sky, visible at j), the target's rendered flow at the SAME queries at beta_rec, and the grader's expected column (median over the "
                                              "sparse X_bg points inside the mask) reproduced", "rows": per_j}
    out["depth_check"] = dep
    od = rep / "m3_s4"; od.mkdir(parents=True, exist_ok=True)
    (od / f"pixel_bridge_{clip}_{cat}{tag}{lift_tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    f = lambda v, n=3: "-" if v is None else f"{v:.{n}f}"                             # noqa: E731
    md = [f"# Pixel bridge decomposed and depth at the same pixels — {args.pair}, {args.frames} frames (reviewer 2026-10-11, checks 2 and 3)", "",
          f"Source motion: CoTracker on the source clip ({cf.name}), frame-0 queries, latent frames 0..{T_v - 1}. Target flow: the render's dense lift re-projected through the "
          f"bridged camera with its translations rescaled to β′ (β_rec = {beta_rec:.3f}); per query patch the median flow of the lifted points in that patch. "
          f"Residual = mean over latent frames 1..{T_v - 1} of ‖u_src − F(β′)‖ in patches. β′ grid {out['beta_grid'][0]:.2f}–{out['beta_grid'][1]:.2f}, {len(betas)} log-spaced.", "",
          f"Queries: frame-0 CoTracker queries inside the render mask with at least {args.min_vis} visible latent frames after frame 0; errors over the visible frames.", "",
          "| region | n | β_pix | β_pix / β_rec | residual at β_pix / at β_rec (patch) | per-patch own-best β′ quartiles | mean displacement over frames 1..8, source / target at β_rec (patch) | target ÷ source at the same pixels (median) |",
          "|---|---|---|---|---|---|---|---|"]
    for name, r in [("overall", out["overall"])] + list(out["by_target_depth_tercile"].items()) + [("contact strip", out["contact_strip"])]:
        qq = r.get("per_query_beta_quartiles")
        md.append(f"| {name} | {r['n']} | {f(r.get('beta_pix'))}{' (grid edge)' if r.get('at_grid_edge') else ''} | {f(r.get('beta_pix_over_beta_rec'))} | "
                  f"{f(r.get('residual_at_beta_pix_patch'), 2)} / {f(r.get('residual_at_beta_rec_patch'), 2)} | {('–'.join(f'{v:.2f}' for v in qq)) if qq else '-'} | "
                  f"{f(r.get('mean_disp_src_patch'), 2)} / {f(r.get('mean_disp_target_at_beta_rec_patch'), 2)} | {f(r.get('target_over_src_at_same_pixels'), 2)} |")
    md += ["", f"Target-depth tercile bounds (MoGe·k_bg, camera t_a): {[round(v, 2) for v in bounds]}; {int(on_strip[idx].sum())} queries on the contact strip.", ""]
    if "n_matched" in dep:
        bt = dep["by_target_depth_tercile"]
        md += [f"Depth at the same pixels: {dep['n_matched']} queries matched to a camera-C held-out point within {args.nn_px:.0f} px (of {dep['n_heldout_points']}).", "",
               "| depth ratio over the covered region | overall | near | mid | far |", "|---|---|---|---|---|",
               f"| MoGe·k_bg / Z_tri | {f(dep['moge_kbg_over_Ztri_median'])} | " + " | ".join(f"{f(bt[k]['over_Ztri'])} (n {bt[k]['n']})" for k in ('near', 'mid', 'far')) + " |",
               f"| MoGe·k_bg / lift | {f(dep['moge_kbg_over_lift_median'])} | " + " | ".join(f(bt[k]['over_lift']) for k in ('near', 'mid', 'far')) + " |",
               f"| lift / Z_tri at the matched points (band ρ under ViPE = {f(rho)}) | {f(dep['lift_over_Ztri_median_at_matched'])} | | | |", "", dep["note"]]
    md += ["", "β_pix per latent frame over the whole covered set (a value that drifts with j is the camera's translation magnitude over time, which no single β fixes):", "",
           "| j | " + " | ".join(str(r["j"]) for r in per_frame) + " |", "|---|" + "---|" * len(per_frame),
           "| β_pix / β_rec | " + " | ".join(f"{r['beta_pix_over_beta_rec']:.2f}" for r in per_frame) + " |",
           "| residual at β_pix / at β_rec | " + " | ".join(f"{r['residual_at_beta_pix']:.2f} / {r['residual_at_beta_rec']:.2f}" for r in per_frame) + " |"]
    md += ["", "β_pix / β_rec per latent frame × target-depth tercile, with σ̂_t / σ̂_{t_a} from the A.5 calibration record over the same frames (reviewer 2026-10-11: "
           "near/mid flat with far ≈ 1 → depth shape; all terciles growing together and tracking σ̂_t → camera drift; both → both, depth first):", "",
           "| j | " + " | ".join(str(j) for j in range(1, T_v)) + " |", "|---|" + "---|" * (T_v - 1)]
    for name in ("near", "mid", "far"):
        md.append(f"| {name} | " + " | ".join("-" if v is None else f"{v:.2f}" for v in frame_tercile[name]) + " |")
    md.append("| σ̂_t / σ̂_t_a | " + (" | ".join(f"{v:.3f}" for v in sigma_ratio) if sigma_ratio else "record not found") + " |")
    g_ = lambda r, k, n=2: "-" if r.get(k) is None else f"{r[k]:.{n}f}"                      # noqa: E731
    md += ["", f"Rotation refit per frame (reviewer §4 arm 1; tracks = {args.tracks}): one point per query at its cell's target depth; per frame a translation scale s alone, then a small rotation "
           "correction δR_t (rotation vector, 3 params) plus s, Huber loss. 1 / s is β′ / β_rec. δθ against the ViPE-vs-D4RT rotation gap of the S-int camera check (every 8 frames, interpolated):", "",
           "| j | " + " | ".join(str(r["j"]) for r in rot_rows) + " |", "|---|" + "---|" * len(rot_rows),
           "| 1/s, scale only | " + " | ".join("-" if r.get("s_scale_only") is None else f"{1 / r['s_scale_only']:.2f}" for r in rot_rows) + " |",
           "| 1/s, with δR_t | " + " | ".join("-" if r.get("s_with_rotation") is None else f"{1 / r['s_with_rotation']:.2f}" for r in rot_rows) + " |",
           "| δθ_t (deg) | " + " | ".join(g_(r, "dtheta_deg") for r in rot_rows) + " |",
           "| ViPE-vs-D4RT gap (deg) | " + (" | ".join(f"{v:.2f}" for v in rot_gap) if rot_gap else "record not found") + " |",
           "| median residual, scale only / with δR_t | " + " | ".join(f"{g_(r, 'resid_scale_only')} / {g_(r, 'resid_with_rotation')}" for r in rot_rows) + " |"]
    for n in ("near", "mid", "far"):
        md.append(f"| 1/s by tercile after δR_t: {n} | " + " | ".join("-" if not r.get("s_by_tercile_after_rotation") or r["s_by_tercile_after_rotation"][n] is None else f"{1 / r['s_by_tercile_after_rotation'][n]:.2f}" for r in rot_rows) + " |")
    md += ["", "Reviewer's reading rule: δR_t grows and absorbs the far growth → ViPE's rotation over the window; growth disappears with D4RT tracks → CoTracker lag; neither → ViPE's translation shape against the source."]
    md += ["", "## The grader's expected column re-taken at the grader's own sampling", "",
           "The grader's expected flow is the median over the sparse X_bg points inside the render mask (a set concentrated in the near field); its measured values are medians over "
           "a uniform pixel grid. Same queries, per latent frame j (render mask, not object, not sky, visible at j):", "",
           "| j | n | source's own tracked displacement | target's rendered flow at the same queries, β_rec | grader's expected (sparse X_bg) |", "|---|---|---|---|---|"]
    for r in per_j:
        md.append(f"| {r['j']} | {r['n']} | {f(r['source_median_patch'], 2)} | {f(r['target_at_same_queries_median_patch'], 2)} | {f(r['grader_expected_sparse_Xbg_patch'], 2)} |")
    md += ["", "Rule (reviewer 2026-10-11): β_pix constant across depth and at the contact → the pixel bridge is the estimator of record (D-M3-08 amendment 2, John signs); "
           "β_pix ≈ β_rec at the contact but larger away from it → the fault is the target's depth shape (§2.6): align MoGe to the source's motion-consistent depth over the covered "
           "region instead of one band scalar, β stays."]
    (od / f"pixel_bridge_{clip}_{cat}{tag}{lift_tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8"); print("\n".join(md)); print("->", od / f"pixel_bridge_{clip}_{cat}{tag}{lift_tag}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
