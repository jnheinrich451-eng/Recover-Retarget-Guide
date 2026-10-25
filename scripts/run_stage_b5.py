#!/usr/bin/env python
"""Stage B.5 runner (SKILL v1.1): heading d_src + scene up u_scene addendum.
CPU-only. Statistics only — Module 2 applies its own gates (§0).

    python scripts/run_stage_b5.py --clip-id <id> [...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml                                            # noqa: E402
from src.preprocess.bundle import read_bundle          # noqa: E402
from src.preprocess.config import load_config          # noqa: E402
from src.preprocess.stage_b.scoring import eta_weights  # noqa: E402
from src.preprocess.stage_c import finalize            # noqa: E402
from src.preprocess.stage_b5 import heading, plane, writer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-id", nargs="*", default=[])
    ap.add_argument("--curated", default=None,
                    help="tier from configs/curated_clips.yaml "
                         "(tier_a_good | tier_b_part_missing | all)")
    ap.add_argument("--config", default="configs/stage_a.yaml")
    ap.add_argument("--config-b5", default="configs/stage_b5.yaml")
    args = ap.parse_args()
    if args.curated:
        cur = yaml.safe_load(Path("configs/curated_clips.yaml")
                             .read_text(encoding="utf-8"))
        tiers = list(cur) if args.curated == "all" else [args.curated]
        args.clip_id = [c for t in tiers for c in cur[t]]
    assert args.clip_id, "give --clip-id ... or --curated <tier>"
    cfg = load_config(args.config)
    b5 = yaml.safe_load(Path(args.config_b5).read_text(encoding="utf-8"))["b5"]
    out_dir = Path(str(cfg["paths"]["drive_root"]) + "/cache/stage_b5")

    for clip_id in args.clip_id:
        b = read_bundle(cfg, clip_id)
        Q = b["Q"]
        F = Q.shape[1]
        boundaries = b["qc"]["window_boundaries"]
        bdir = Path(cfg["paths"]["stage_b_cache"])
        qc_b = json.loads((bdir / f"{clip_id}_qc_b.json").read_text(encoding="utf-8"))
        with np.load(bdir / f"{clip_id}.npz") as z:
            sb = {k: z[k] for k in z.files}
        M, present, t_c = finalize.read_mask(
            Path(cfg["paths"]["stage_c_cache"]) / clip_id)
        eta = eta_weights(b["X0"], b["X_local"], b["G"], Q)
        segs = qc_b["segment_scales"]
        fcr = qc_b.get("follow_cam") or {}
        branch = bool(fcr.get("branch_active"))
        gcurve = heading.seg_scale_curve(segs, F)

        # dense union when the C.5 addendum exists
        subject = sb["fc_blob_idx"] if branch else sb["o_src_members"]
        c5p = Path(str(cfg["paths"]["drive_root"]) + "/cache/stage_c5") \
            / f"{clip_id}.npz"
        Xl_all, Q_all = b["X_local"], Q
        if c5p.exists():
            with np.load(c5p) as z:
                Xl_all = np.concatenate([b["X_local"], z["X_local_dense"]])
                Q_all = np.concatenate([Q, z["Q_dense"]])
                subject = np.concatenate(
                    [subject, np.arange(len(Q), len(Q_all))])

        # centroid track (channel-dispatched, §2.1)
        traj_source = str(b5.get("traj_source", "G"))
        estimator = str(b5.get("heading_estimator", "v1.2"))
        L, L_min = int(b5["heading_L"]), int(b5["heading_L_min"])
        v12 = None
        if branch:
            assert traj_source == "G", \
                "vipe traj_source not wired yet — measured comparison pending"
            ts = int(fcr["seed_frame"])
            mem = subject[Q_all[subject, ts]]
            c_cam_ts = np.nanmean(Xl_all[mem, ts].astype(np.float64), axis=0)
            xb, branch_ratio = heading.centroid_track_branch(
                b["G"].astype(np.float64), c_cam_ts,
                sb["fc_tau_rel"].astype(np.float64), sb["fc_rel_valid"])
            xb[~present] = np.nan
            d_src, sigma_theta, wrec = heading.wls_heading(
                xb, ts, L, L_min, boundaries, segs)
            est_used = "branch_composed"
        else:
            ts = t_c
            branch_ratio = float("nan")
            if estimator == "v1.2":
                # common-support unweighted centroid (Gate A 2026-09-10);
                # X_bar_track stays the per-frame v1.1 track = position channel
                v12 = heading.heading_v12(
                    b["X0"].astype(np.float64), Q, b["P"], M,
                    sb["o_src_members"], eta, present,
                    b["G"].astype(np.float64), t_c, L, L_min, boundaries,
                    segs, irls_passes=int(b5.get("irls_passes", 3)))
                xb = v12["X_bar_v11"]
                d_src, sigma_theta, wrec = v12["d_src"], v12["sigma_theta"], v12["window"]
                est_used = "v1.2_common_support_unweighted"
            elif estimator == "v1.1":
                xb = heading.centroid_track_x0(
                    b["X0"].astype(np.float64), Q, b["P"], M,
                    sb["o_src_members"], eta)
                xb[~present] = np.nan
                d_src, sigma_theta, wrec = heading.wls_heading(
                    xb, ts, L, L_min, boundaries, segs)
                est_used = "v1.1_per_frame_eta"
            else:
                raise ValueError(f"unknown heading_estimator {estimator!r}")

        # scene up (§2.3; branch pool excludes the blob — v1.1 delta 3)
        excl = sb["fc_blob_idx"] if (branch and "fc_blob_idx" in sb) else None
        Zmed = float(np.nanmedian(b["X_local"][..., 2][Q]))
        pts = plane.static_pool(b["X0"].astype(np.float64), Q, sb["y_dyn"],
                                eta, gcurve, exclude=excl,
                                eta_q=float(b5["eta_min_quantile"]))
        pl = plane.fit_plane(pts, iters=int(b5["ransac_iters"]),
                             tol=float(b5["ransac_tol_frac"]) * Zmed) \
            if len(pts) >= int(b5["min_plane_points"]) else None
        cen = np.nanmean(xb[np.isfinite(xb[:, 0])], axis=0) \
            if np.isfinite(xb[:, 0]).any() else np.zeros(3)
        if pl is not None:
            n, u_b, ang = plane.orient_and_cross(pl, cen, b["G"], int(ts))
            u_scene, up_source = n, "plane"
        else:
            u_b = -b["G"][int(ts), :3, :3] @ np.array([0.0, 1.0, 0.0])
            u_b /= np.linalg.norm(u_b)
            n, u_scene, ang, up_source = None, u_b, 0.0, "camera_prior"

        kappa = (float(np.linalg.norm(np.cross(u_scene, d_src)))
                 if d_src is not None else float("nan"))
        rec = {"clip_id": clip_id, "schema": "b5_v1.2", "plane_d_signed_with_n": True,
               "channel": "branch" if branch else "x0",
               "heading_estimator": est_used,
               "traj_source": traj_source if branch else None,
               "heading_defined": bool(d_src is not None),
               "sigma_theta": float(sigma_theta),
               "heading_window": wrec, "up_source": up_source,
               "planarity": None if pl is None else pl["planarity"],
               "spread": None if pl is None else pl["spread"],
               "n_pool": int(len(pts)),
               "angle_ua_ub_deg": float(ang),
               "branch_ratio": float(branch_ratio),
               "kappa_basis": float(kappa),
               "up_defined": bool(up_source != "undefined"),
               "frames": {"d_src": "camera_0", "u_scene": "camera_0",
                          "X_bar_track": "camera_0", "rho_*": "camera_t_a"}}
        arrays = {"X_bar_track": xb.astype(np.float64),
                  "u_scene": np.asarray(u_scene, np.float64)}
        if v12 is not None:
            # v1.2 statistics (contract_stage_b5_v1_2 §3); absent on branch
            rec.update({"n_star": v12["n_star"], "n_union": v12["n_union"],
                        "support_frac": v12["support_frac"],
                        "per_frame_counts": v12["per_frame_counts"],
                        "churn_deg": v12["churn_deg"],
                        "sigma_spec_deg": v12["sigma_spec_deg"],
                        "mad_convention": "1.4826*MAD"})
            if v12["ident"] is not None:
                rec.update({k: v12["ident"][k] for k in
                            ("rho_radial", "rho_lat1", "rho_lat2", "median_v_radial")})
                rec["rho_frame"] = v12["ident"]["frame"]
            if v12["X_bar_common"] is not None:
                arrays["X_bar_common"] = v12["X_bar_common"].astype(np.float64)
            if v12["d_v11"] is not None:
                arrays["d_v11"] = np.asarray(v12["d_v11"], np.float64)
        if d_src is not None:
            arrays["d_src"] = np.asarray(d_src, np.float64)
        if pl is not None:
            arrays["plane_n"] = np.asarray(n, np.float64)
            arrays["plane_d"] = np.asarray([pl["d"]], np.float64)        # re-signed with n by orient_and_cross (2026-09-14)
        # hash chain: this stage's config + every upstream artifact read
        qc_c_path = Path(cfg["paths"]["stage_c_cache"]) / clip_id / "qc_c.json"
        qc_c = json.loads(qc_c_path.read_text(encoding="utf-8")) if qc_c_path.exists() else {}
        hashes = {"config": cfg["_meta"]["config_hash"],
                  "bundle": b["provenance"].get("config_hash"),
                  "stage_b5_config": hashlib.sha256(
                      Path(args.config_b5).read_bytes()).hexdigest()[:16],
                  "stage_b": qc_b.get("config_hash"),
                  "stage_c": qc_c.get("config_hash")}
        writer.write_b5(out_dir, clip_id, rec, arrays, hashes)
        extra = (f" churn={v12['churn_deg']:.1f}deg n*={v12['n_star']}/{v12['n_union']} "
                 f"spec={v12['sigma_spec_deg']:.1f}deg" if v12 is not None else "")
        print(f"{clip_id}: {rec['channel']} [{est_used}] heading="
              f"{'-' if d_src is None else np.round(d_src, 3).tolist()} "
              f"sigma_theta={sigma_theta:.4f} up={up_source} "
              f"pi={rec['planarity']} ratio={branch_ratio:.3f} "
              f"kappa={kappa:.3f}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
