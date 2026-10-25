#!/usr/bin/env python
"""What the unguided A0 videos do, in numbers (the A0 row of S4, and the instrument that grades every guided run the same way):
CoTracker3 on the generated video from a grid at frame 0; tracks split into object (inside the painted anchor mask at frame 0) and
background; per latent frame:  background median displacement (patches)  = the camera the model chose;
                                object centroid displacement (patches)     = where the model moved the object;
                                object deformation = median |relative displacement of object tracks about their centroid| / object size
                                (0 = rigid translation; a seed that "renders the motion into the object" reads high);
against the pair's geometric target from the geometry cache: expected background flow (X_bg through E'_t with the bridge) and the
expected object centroid path. Phase 2 (CoTracker needs the GPU). Writes reports/m3_s4/a0_motion.{json,md}; rule 24.

    python scripts/measure_m3_a0.py --pairs 000000000265.2.005/vehicle 000000000265.2.005/human
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.probe_cotracker import read_video                                   # noqa: E402
from scripts.run_m3_s2_readout import WAN_H, WAN_W                               # noqa: E402
from src.generate.m3 import guidance as GD
from src.generate.m3 import render_target as RT
from src.preprocess.stage_c import finalize                                       # noqa: E402
from src.generate.m3 import handoff as H                                          # noqa: E402
from src.generate.m3.geometry import project as PJ                               # noqa: E402



def source_video_path(videos_dir, clip, n_frames_expected=None, sha256_expected=None):
    """The source video of record for a clip (reviewer 2026-10-20: asserted by CONTENT, never by lookup order). Candidates: the pool
    subfolders and the flat file under <videos_dir>; the one whose frame count equals the record's (the Stage C mask length) and, when the
    set carries a hash (configs/experiment_video_hashes.yaml), whose sha256 equals it. Exactly one candidate may satisfy that."""
    import hashlib
    cands = sorted(Path(videos_dir).glob(f"*/{clip}.mp4")) + ([Path(videos_dir) / f"{clip}.mp4"] if (Path(videos_dir) / f"{clip}.mp4").exists() else [])
    assert cands, f"source video {clip}.mp4 not found under {videos_dir} (a pool subfolder or flat)"
    if n_frames_expected is None and sha256_expected is None:
        assert len(cands) == 1, f"source video {clip}.mp4 is ambiguous under {videos_dir} and no frame count / hash was given: {cands}"
        return cands[0]
    ok = []
    for c in cands:
        cap = cv2.VideoCapture(str(c)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        h = hashlib.sha256(c.read_bytes()).hexdigest() if sha256_expected else None
        if sha256_expected is not None:
            if h == sha256_expected:
                ok.append((c, n, h))
        elif n_frames_expected is None or n >= int(n_frames_expected):           # Stage A consumed the FIRST n_frames_expected frames: a longer file (the six pairs' originals) qualifies
            ok.append((c, n, h))
    if len(ok) > 1 and n_frames_expected is not None:
        exact = [o for o in ok if o[1] == int(n_frames_expected)]
        ok = exact or ok
    assert len(ok) == 1, f"source video {clip}.mp4: {len(ok)} candidates match frames >= {n_frames_expected} / hash {sha256_expected and sha256_expected[:12]} under {videos_dir}: {[str(c) for c in cands]}"
    return ok[0][0]


def expected_source(root, clip, cat):
    """(frame count, sha256 or None) of the clip's source of record: the handoff's n_frames_source and the set's recorded hash when present."""
    h = json.loads((Path(root) / "cache" / "m2_e" / clip / cat / "handoff.json").read_text(encoding="utf-8"))
    hf = ROOT / "configs" / "experiment_video_hashes.yaml"
    hashes = (yaml.safe_load(hf.read_text(encoding="utf-8")) or {}).get("sha256", {}) if hf.exists() else {}
    return int(h["n_frames_source"]), hashes.get(clip)

def track(cot, video_u8, grid_px: int, dev):
    import torch
    vt = torch.from_numpy(video_u8).permute(0, 3, 1, 2)[None].float().to(dev)      # [1,F,3,H,W]
    ys, xs = np.mgrid[grid_px // 2:WAN_H:grid_px, grid_px // 2:WAN_W:grid_px]
    q = np.stack([np.zeros(xs.size), xs.ravel(), ys.ravel()], 1).astype(np.float32)   # (t, x, y)
    with torch.no_grad():
        tr, vis = cot(vt, queries=torch.from_numpy(q)[None].to(dev))
    return tr[0].cpu().numpy(), vis[0].cpu().numpy().astype(bool), np.stack([xs.ravel(), ys.ravel()], 1)


def similarity_2d(p0, p1):
    """Translation + isotropic scale from points p0 [n,2] to p1 [n,2] (least squares): s = sqrt(var1 / var0), t = mean1 - s * mean0."""
    c0, c1 = p0.mean(0), p1.mean(0)
    v0 = ((p0 - c0) ** 2).sum(1).mean(); v1 = ((p1 - c1) ** 2).sum(1).mean()
    s_ = float(np.sqrt(v1 / max(v0, 1e-9)))
    return s_, (c1 - s_ * c0)


def summarise(tracks, vis, q0, obj_mask_wan, px, T_l, sky_wan=None, region_wan=None, F_expected=None, source=None, min_src_patch=0.25, F_obj_expected=None, outside_split=None,
              source_dyn=None, nonrigid_region=None, ground_wan=None, coverage_split=None):
    """tracks [F,N,2] px, vis [F,N]; obj_mask_wan [480,832] bool at frame 0. Per latent frame j (video frame 4j). Background = not object and
    not sky. With region_wan [480,832] (the render mask at frame 0), the background displacement is also given inside / outside that region:
    a partial correction moves the covered part and leaves the rest to the model, which a whole-frame median cannot see."""
    qi, qj = q0[:, 1].astype(int).clip(0, WAN_H - 1), q0[:, 0].astype(int).clip(0, WAN_W - 1)
    inobj = obj_mask_wan[qi, qj]
    insky = sky_wan[qi, qj] if sky_wan is not None else np.zeros(len(q0), bool)
    inreg = region_wan[qi, qj] if region_wan is not None else None
    # F_expected [T_l, Q, 2] px: the target's flow at these same queries (NaN where undefined); the expected value is then the median over the
    # tracks that are visible at frame j, exactly the set the measured median uses (a lost track is dropped from both)
    fex = np.isfinite(F_expected).all(2) if F_expected is not None else None
    # F_obj_expected [T_l, Q, 2] px: the RENDER's object flow at these queries (the anchor's object pixels moved along the path). The expected
    # centroid is then taken over the same surviving object tracks as the measured one (reviewer 2026-10-11 §1: lost tracks sit at the smeared
    # rear and shift the survivors' centroid), and a 2-D similarity (scale + translation) frame-0 -> frame-f is fitted to both.
    fox = np.isfinite(F_obj_expected).all(2) if F_obj_expected is not None else None
    # outside_split: dict name -> [Q] bool over queries (e.g. uncovered at frame 0 / covered but excluded / sky): the outside-mask figures per part
    # source = (tracks, vis) of the source clip on the same query grid: the direction of the outside-mask motion is read against it, over the
    # outside background tracks visible in both at frame f whose source displacement exceeds min_src_patch (a near-zero vector has no direction)
    if source is not None:
        assert source[0].shape[1] == len(q0) and np.allclose(source[0][0], q0), "the source must be tracked from the same query grid"
    # source_dyn [F, Q] bool: the source track lies inside the SOURCE object's dilated dynamic mask at frame f (its legs, shadow, dust): such a
    # track is not background motion and is left out of the direction reference (reviewer 2026-10-11)
    sdyn = source_dyn if source_dyn is not None else (np.zeros(source[0].shape[:2], bool) if source is not None else None)
    size = np.sqrt(obj_mask_wan.sum()) / px                                       # object size in patches (sqrt area)
    out = []
    for j in range(T_l):
        f = 4 * j
        d = (tracks[f] - tracks[0]) / px
        vb, vo = vis[f] & ~inobj & ~insky, vis[f] & inobj
        bg = np.linalg.norm(d[vb], axis=1)
        rec = {"latent_frame": j, "bg_disp_median_patch": float(np.median(bg)) if vb.sum() else None, "bg_disp_p90_patch": float(np.quantile(bg, 0.9)) if vb.sum() else None,
               "bg_mean_vector_patch": d[vb].mean(0).tolist() if vb.sum() else None, "n_bg": int(vb.sum()), "n_obj": int(vo.sum()),
               **({"bg_in_region_expected_at_visible_patch": float(np.median(np.linalg.norm(F_expected[j][vb & inreg & fex[j]], axis=1)) / px) if (vb & inreg & fex[j]).sum() >= 5 else None}
                  if (inreg is not None and fex is not None) else {}),
               **({"bg_in_region_median_patch": float(np.median(np.linalg.norm(d[vb & inreg], axis=1))) if (vb & inreg).sum() >= 5 else None,
                   "bg_out_region_median_patch": float(np.median(np.linalg.norm(d[vb & ~inreg], axis=1))) if (vb & ~inreg).sum() >= 5 else None,
                   "n_bg_in_region": int((vb & inreg).sum())} if inreg is not None else {}),
               "obj_track_survival": float(vo.sum() / max(inobj.sum(), 1))}          # share of frame-0 object tracks still visible: 0 = the object is gone (seed 1's signature)
        if fox is not None and (vo & fox[j]).sum() >= 4:
            vq_ = vo & fox[j]
            ce = F_obj_expected[j][vq_].mean(0) / px
            s_m, t_m = similarity_2d(tracks[0][vq_], tracks[f][vq_]); s_e, t_e = similarity_2d(q0[vq_], q0[vq_] + F_obj_expected[j][vq_])
            rec.update({"obj_centroid_expected_at_survivors_patch": ce.tolist(), "obj_centroid_expected_at_survivors_norm": float(np.linalg.norm(ce)), "n_obj_survivors_with_expectation": int(vq_.sum()),
                        "obj_similarity_measured": {"scale": s_m, "t_patch": (t_m / px).tolist()}, "obj_similarity_render": {"scale": s_e, "t_patch": (t_e / px).tolist()}})
        if nonrigid_region is not None and fex is not None:
            inr_ = nonrigid_region[qi, qj] & vb & fex[j]
            if inr_.sum() >= 5:
                resid_ = d[inr_] - F_expected[j][inr_] / px                                 # what moved beyond the camera's rigid prediction
                rec["nonrigid_region_resid_median_patch"] = float(np.median(np.linalg.norm(resid_, axis=1)))
                rec["nonrigid_region_resid_p90_patch"] = float(np.quantile(np.linalg.norm(resid_, axis=1), 0.9)); rec["n_nonrigid_region"] = int(inr_.sum())
        if outside_split is not None and inreg is not None:
            parts = {}
            for name, part in outside_split.items():
                sel_ = vb & ~inreg & part
                entry = {"n": int(sel_.sum()), "median_patch": float(np.median(np.linalg.norm(d[sel_], axis=1))) if sel_.sum() >= 5 else None}
                if source is not None and f < source[0].shape[0]:
                    ds_ = (source[0][f] - source[0][0]) / px
                    both_ = sel_ & (source[1][f] > 0.5) & (np.linalg.norm(ds_, axis=1) >= min_src_patch) & ~sdyn[f]
                    if both_.sum() >= 5:
                        cs_ = (d[both_] * ds_[both_]).sum(1) / (np.linalg.norm(d[both_], axis=1) * np.linalg.norm(ds_[both_], axis=1) + 1e-9)
                        entry.update({"cos_vs_source_median": float(np.median(cs_)), "n_cos": int(both_.sum())})
                parts[name] = entry
            rec["outside_parts"] = parts
        if coverage_split is not None:                                                 # reviewer 2026-10-14: classes defined on coverage, independent of the mask
            rec["track_classes"] = {k: int((vis[f] & m_).sum()) for k, m_ in coverage_split.items()}
            rec["track_classes"]["object"] = int(vo.sum())
            if source is not None and f < source[0].shape[0]:                          # direction against the source per coverage class, same rule as the outside cosine
                ds_c = (source[0][f] - source[0][0]) / px
                for k, m_ in coverage_split.items():
                    both_c = vis[f] & m_ & (source[1][f] > 0.5) & (np.linalg.norm(ds_c, axis=1) >= min_src_patch) & ~sdyn[f]
                    if both_c.sum() >= 5:
                        cs_c = (d[both_c] * ds_c[both_c]).sum(1) / (np.linalg.norm(d[both_c], axis=1) * np.linalg.norm(ds_c[both_c], axis=1) + 1e-9)
                        rec["track_classes"][f"{k}_cos_vs_source"] = float(np.median(cs_c)); rec["track_classes"][f"{k}_n_cos"] = int(both_c.sum())
                        rec["track_classes"][f"{k}_median_patch"] = float(np.median(np.linalg.norm(d[both_c], axis=1)))
            if inreg is not None:
                rec["track_classes"].update({"in_mask": int((vb & inreg).sum()), "out_of_mask": int((vb & ~inreg).sum())})
        if source is not None and inreg is not None and f < source[0].shape[0]:
            ds = (source[0][f] - source[0][0]) / px
            both = vb & ~inreg & (source[1][f] > 0.5) & (np.linalg.norm(ds, axis=1) >= min_src_patch) & ~sdyn[f]
            if both.sum() >= 5:
                cs = (d[both] * ds[both]).sum(1) / (np.linalg.norm(d[both], axis=1) * np.linalg.norm(ds[both], axis=1) + 1e-9)
                rec.update({"outside_cos_vs_source_median": float(np.median(cs)), "n_outside_cos": int(both.sum())})
        if vo.sum() >= 4:
            c = d[vo].mean(0); rel = np.linalg.norm(d[vo] - c, axis=1)
            rec.update({"obj_centroid_disp_patch": c.tolist(), "obj_centroid_disp_norm": float(np.linalg.norm(c)), "obj_deformation": float(np.median(rel) / max(size, 1e-6)),
                        "obj_scale_change": float(np.sqrt(((tracks[f][vo] - tracks[f][vo].mean(0)) ** 2).sum(1).mean() / max(((tracks[0][vo] - tracks[0][vo].mean(0)) ** 2).sum(1).mean(), 1e-9)))})
        out.append(rec)
    # full-rate object centroid (tracks visible at every frame) detrended by Savitzky-Golay (13, 2): the gait-period oscillation (reviewer 2026-10-11 decision 2)
    vo_all = inobj & vis.all(0)
    if vo_all.sum() >= 4 and tracks.shape[0] >= 15:
        from scipy.signal import savgol_filter
        cen = tracks[:, vo_all].mean(1) / px
        resid = cen - savgol_filter(cen, 13, 2, axis=0)
        out[-1]["obj_centroid_gait_oscillation_patch"] = float(np.linalg.norm(resid, axis=1).max())
        out[-1]["obj_centroid_gait_oscillation_rms_patch"] = float(np.sqrt((resid ** 2).sum(1).mean()))
        # relative to the local background: background tracks (not object, not sky) within local_px of the object's frame-0 centroid, visible at
        # every frame; their mean displacement is subtracted from the object centroid before detrending (reviewer 2026-10-12)
        c0 = tracks[0][vo_all].mean(0); local_px = 6 * px
        near = (~inobj) & (~insky) & vis.all(0) & (np.linalg.norm(q0 - c0, axis=1) <= local_px)
        if ground_wan is not None:                                                     # reviewer 2026-10-14: the ground is the rigid part only (rock, not the river)
            near = near & ground_wan[qi, qj]
        out[-1]["local_ground"] = "ground mask" if ground_wan is not None else "all non-object, non-sky tracks within 6 patches"
        if near.sum() >= 8:
            bg_local = tracks[:, near].mean(1) / px
            rel = cen - bg_local
            resid_rel = rel - savgol_filter(rel, 13, 2, axis=0)
            out[-1]["obj_centroid_gait_oscillation_rel_patch"] = float(np.linalg.norm(resid_rel, axis=1).max())
            out[-1]["n_local_bg_tracks"] = int(near.sum())
            out[-1]["obj_rel_displacement_last_patch"] = float(np.linalg.norm(rel[-1] - rel[0]))        # locomotion: the object's motion against its local background
            out[-1]["obj_rel_path_length_patch"] = float(np.linalg.norm(np.diff(rel, axis=0), axis=1).sum())
    return out, float(size)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", default=__import__("os").environ.get("MT_ROOT", "/content/drive/MyDrive/motion_transfer"))
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--grid-px", type=int, default=16)
    ap.add_argument("--videos", nargs="*", default=None, help="video files to grade (default: every A0_seed*.mp4 of the pair); 'source' adds the source clip from t_a")
    ap.add_argument("--videos-dir", default=__import__("os").environ.get("MT_DATA", "/content/data") + "/videos")
    ap.add_argument("--source-frames", type=int, default=None, help="grade 'source' over this many frames (default: the pair's full window); 33 to match a first-light video")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--ground-mask", default=None, help="PNG (white = rigid ground, e.g. the rocks = NOT the water mask): the local ground reference for the person's locomotion and gait-rel "
                    "is restricted to tracks inside it (reviewer 2026-10-14)")
    ap.add_argument("--region-mask", default=None, help="PNG (white = region, e.g. the water) in which the NON-RIGID residual is reported: track displacement minus the "
                    "camera's rendered flow at the query (reviewer 2026-10-13 check 1: who froze the water, A0 vs A3)")
    ap.add_argument("--dyn-dilate-px", type=int, default=16, help="dilation of the source object's Stage C mask when excluding source tracks from the outside-direction reference")
    args = ap.parse_args(argv)
    import cv2
    import torch
    lock = H.load_m3_lock(); px = int(lock["lattice"]["px_per_patch"]); Hp, Wp = int(lock["lattice"]["Hp"]), int(lock["lattice"]["Wp"])
    dev = "cuda"; root = Path(args.drive_root)
    cot = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(dev).eval()
    rows = []
    for pair in args.pairs:
        clip, cat = pair.split("/")
        d = root / "cache" / "m3" / clip / cat
        with np.load(d / "lift.npz") as z:
            amask_sq, X_bg, Kp = z["anchor_mask_sq"].astype(bool), z["X_bg"].astype(np.float64), z["Kp"].astype(np.float64)
            Dsq, msq, Kq, K_wan = z["depth_src_sq"].astype(np.float64), z["mask_sq"].astype(bool), z["K_sq"].astype(np.float64), z["K_wan"].astype(np.float64)
            z_sky = z["sky_lattice"].astype(bool) if "sky_lattice" in z.files else None
        if z_sky is not None:
            z_sky = cv2.resize(z_sky.astype(np.uint8), (WAN_W, WAN_H), interpolation=cv2.INTER_NEAREST).astype(bool)
        obj_wan = cv2.resize(amask_sq.astype(np.uint8), (WAN_W, WAN_H), interpolation=cv2.INTER_NEAREST).astype(bool)
        Xd, _ = RT.lift_dense(Dsq, msq & ~amask_sq, Kq, np.zeros(Dsq.shape + (3,), np.uint8))   # the render's dense lift (background only)
        Xo_d, _ = RT.lift_dense(Dsq, msq & amask_sq, Kq, np.zeros(Dsq.shape + (3,), np.uint8))   # the object's visible surface at frame 0 (what the object render moves)
        covered0 = None; covered_t = None
        if (d / "render.npz").exists():
            with np.load(d / "render.npz") as z:
                covered0 = z["covered_px0"].astype(bool) if "covered_px0" in z.files else None   # renders made before 2026-10-11 carry no coverage map
                covered_t = z["covered_px"].astype(bool) if "covered_px" in z.files else None    # per frame: the newly revealed strips exist only at t > 0
        _exp_cache = {}

        def expectations(tag):
            """The geometric target under the bridge `tag` (rec | mid | far): a video is read against the geometry of its own bridge."""
            if tag in _exp_cache:
                return _exp_cache[tag]
            with np.load(d / ("geometry.npz" if tag == "rec" else f"geometry_beta_{tag}.npz")) as z:
                E_l, c_tar = z["E_l"].astype(np.float64), z["c_tar"].astype(np.float64)
                E_full = z["E_full"].astype(np.float64)
                beta_tag = str(z["bridge_run"]) if "bridge_run" in z.files else "v2.5"
            T_l = len(E_l)
            # the geometric target: background flow of X_bg under E'_t (median per latent frame) and the object centroid path on the lattice
            pb, zz = PJ.project(X_bg, E_l, Kp)
            ok0 = (zz[0] > 0) & (pb[0, :, 0] >= 0) & (pb[0, :, 0] < Wp) & (pb[0, :, 1] >= 0) & (pb[0, :, 1] < Hp)
            exp_bg = [float(np.median(np.linalg.norm((pb[j] - pb[0])[ok0 & (zz[j] > 0)], axis=1))) for j in range(T_l)]
            rp = d / "render.npz"
            region = None; exp_in = None; exp_in_dense = None
            if rp.exists():
                with np.load(rp) as z:
                    region = z["mask_px"][0].astype(bool)                                   # the render mask at frame 0: where a render correction applies
                pin = region[np.clip((pb[0, :, 1] * px).astype(int), 0, WAN_H - 1), np.clip((pb[0, :, 0] * px).astype(int), 0, WAN_W - 1)]
                exp_in = [float(np.median(np.linalg.norm((pb[j] - pb[0])[ok0 & (zz[j] > 0) & pin], axis=1))) if (ok0 & pin).sum() >= 5 else None for j in range(T_l)]
                # the same expectation at the grader's own query grid from the render's dense lift (the sparse-X_bg median above over-weights the near
                # field: 3.84 vs 3.08 at j = 8 on 265.2.005/vehicle, pixel_bridge check 2026-10-11); this is the column a measured value is read against
                ysg, xsg = np.mgrid[args.grid_px // 2:WAN_H:args.grid_px, args.grid_px // 2:WAN_W:args.grid_px]
                qg = np.stack([xsg.ravel(), ysg.ravel()], 1).astype(np.float64)
                Fd, _ = RT.flow_at_queries(Xd, E_full[::4][:T_l], K_wan, qg, WAN_H, WAN_W, cell_px=args.grid_px)
                qgi, qgj = qg[:, 1].astype(int), qg[:, 0].astype(int)
                bg_q = region[qgi, qgj] & ~obj_wan[qgi, qgj] & ~(z_sky[qgi, qgj] if z_sky is not None else False) & np.isfinite(Fd).all((0, 2))
                exp_in_dense = [float(np.median(np.linalg.norm(Fd[j][bg_q], axis=1)) / px) if bg_q.sum() >= 5 else None for j in range(T_l)]
            pc, _ = PJ.project(c_tar[:, None, :], E_l, Kp)                              # c_tar [T,3] -> [T,1,3]: per-frame projection of the path
            exp_obj = [float(np.linalg.norm(pc[j, 0] - pc[0, 0])) for j in range(T_l)]
            _exp_cache[tag] = dict(E_l=E_l, E_full=E_full, c_tar=c_tar, beta_tag=beta_tag, T_l=T_l, region=region, exp_bg=exp_bg, exp_in=exp_in, exp_in_dense=exp_in_dense, exp_obj=exp_obj)
            return _exp_cache[tag]
        _e0 = expectations("rec"); T_l = _e0["T_l"]; region = _e0["region"]
        vids = [Path(v) if (v == "source" or Path(v).is_absolute() or Path(v).exists()) else d / v for v in args.videos] if args.videos else sorted(d.glob("A0_seed*.mp4"))   # bare names live in the pair's cache folder
        t_a = int(json.loads((root / "cache" / "m2_e" / clip / cat / "handoff.json").read_text(encoding="utf-8"))["t_a"])
        sky_wan = cv2.resize(z_sky.astype(np.uint8), (WAN_W, WAN_H), interpolation=cv2.INTER_NEAREST).astype(bool) if z_sky is not None else np.zeros((WAN_H, WAN_W), bool)
        if any(str(v) == "source" for v in vids):                                    # the source first: its tracks give the outside-mask direction
            vids = [v for v in vids if str(v) == "source"] + [v for v in vids if str(v) != "source"]
        src_tracks = None; src_dyn = None
        region_mask = None; ground_wan = None
        if args.ground_mask:
            gm_ = cv2.imread(args.ground_mask, cv2.IMREAD_GRAYSCALE); assert gm_ is not None, f"{args.ground_mask} unreadable"
            ground_wan = cv2.resize(gm_, (WAN_W, WAN_H), interpolation=cv2.INTER_NEAREST) > 127
        if args.region_mask:
            rm_ = cv2.imread(args.region_mask, cv2.IMREAD_GRAYSCALE); assert rm_ is not None, f"{args.region_mask} unreadable"
            region_mask = cv2.resize(rm_, (WAN_W, WAN_H), interpolation=cv2.INTER_NEAREST) > 127
        Msrc = None
        try:
            Msrc, _, _ = finalize.read_mask(root / "cache" / "stage_c" / clip)                 # the source object's mask per frame (Stage C)
        except Exception as e_:                                                                 # noqa: BLE001
            print(f"[grader] no Stage C mask for {clip} ({e_}): the outside-direction reference keeps every source track", flush=True)
        for vp in vids:
            vtag = "far" if "_far_" in Path(vp).name else "mid" if "_mid_" in Path(vp).name else "rec"
            rsuf = ("_obj" if "_obj" in Path(vp).name else "") + ("_sky" if "_sky" in Path(vp).name else "") + ("_nodyn" if "_nodyn" in Path(vp).name else "")
            region = _e0["region"]; cov_run_t = covered_t; render_cont = None
            rfile = d / (f"render{rsuf}.npz" if rsuf else "render.npz")
            if rfile.exists():
                with np.load(rfile) as zr:
                    region = zr["mask_px"][0].astype(bool)                                  # this run's own correction mask at frame 0
                    if "covered_px" in zr.files:
                        cov_run_t = zr["covered_px"].astype(bool)                            # this run's own coverage per frame (the sky render covers the sky)
                    rf_ = zr["frames"]
                    rd_ = [float(np.abs(rf_[i + 1].astype(np.float32) - rf_[i].astype(np.float32)).mean()) for i in range(min(len(rf_), 4 * (T_l - 1) + 1) - 1)]
                    render_cont = {"first_step": rd_[0], "median_step": float(np.median(rd_[1:])), "first_over_median": float(rd_[0] / max(np.median(rd_[1:]), 1e-6))}
                print(f"[grader] {vp.name}: inside/outside split from {rfile.name} (mask share {region.mean():.3f}); render continuity first/median {render_cont['first_step']:.2f}/{render_cont['median_step']:.2f}", flush=True)
            ex = expectations(vtag); E_full, c_tar, beta_tag = ex["E_full"], ex["c_tar"], ex["beta_tag"]
            exp_bg, exp_in, exp_in_dense, exp_obj = ex["exp_bg"], ex["exp_in"], ex["exp_in_dense"], ex["exp_obj"]
            if str(vp) == "source":
                n_src = args.source_frames or (4 * (T_l - 1) + 1)
                assert (n_src - 1) % 4 == 0 and n_src <= 4 * (T_l - 1) + 1, n_src
                n_exp, sha_exp = expected_source(root, clip, cat); src_path = source_video_path(args.videos_dir, clip, n_exp, sha_exp)
                video = read_video(src_path, t_a + n_src, WAN_W, WAN_H)[t_a: t_a + n_src]; vp = Path("source")
            else:
                assert vp.exists(), f"{vp} does not exist: the run that writes it has not completed (a sampler assert, e.g. a missing lock key, leaves no video)"
                n_v = int(cv2.VideoCapture(str(vp)).get(cv2.CAP_PROP_FRAME_COUNT))   # the video's own length: a 33-frame first-light video is graded over its 9 latent frames
                video = read_video(vp, min(n_v, 4 * (T_l - 1) + 1), WAN_W, WAN_H)
            T_v = 1 + (len(video) - 1) // 4
            assert len(video) == 4 * (T_v - 1) + 1 and T_v <= T_l, (vp, len(video), T_l)
            fd_ = [float(np.abs(video[i + 1].astype(np.float32) - video[i].astype(np.float32)).mean()) for i in range(len(video) - 1)]
            continuity = {"first_step": fd_[0], "median_step": float(np.median(fd_[1:])), "max_step": float(np.max(fd_)), "first_over_median": float(fd_[0] / max(np.median(fd_[1:]), 1e-6))}
            tr, vis, q0 = track(cot, video, args.grid_px, dev)
            F_exp = None; F_obj = None; split = None; cov_split = None
            if region is not None:
                Fq_, _ = RT.flow_at_queries(Xd, E_full[::4][:T_v], K_wan, q0, WAN_H, WAN_W, cell_px=args.grid_px)   # the target's flow at this video's queries
                F_exp = Fq_
                shift_l = c_tar[:T_v] - c_tar[0]                                                                       # the object's path, translation only (R_t = I)
                F_obj, _ = RT.flow_at_queries(Xo_d, E_full[::4][:T_v], K_wan, q0, WAN_H, WAN_W, cell_px=args.grid_px, shift=shift_l)
                if covered0 is not None:
                    qi_, qj_ = q0[:, 1].astype(int).clip(0, WAN_H - 1), q0[:, 0].astype(int).clip(0, WAN_W - 1)
                    # a query is "newly revealed" when the render leaves it uncovered at the LAST graded frame (the strips the camera motion opens);
                    # "covered but excluded" when the render covers it there but the mask leaves it out (object band, erosion ring)
                    cov_last = cov_run_t[4 * (T_v - 1)] if cov_run_t is not None else covered0
                    split = {"uncovered_at_last": ~cov_last[qi_, qj_] & ~sky_wan[qi_, qj_], "covered_but_excluded": cov_last[qi_, qj_] & ~sky_wan[qi_, qj_], "sky": sky_wan[qi_, qj_]}
                    nobj_ = ~obj_wan[qi_, qj_]
                    cov_split = {"covered": cov_last[qi_, qj_] & ~sky_wan[qi_, qj_] & nobj_, "uncovered": ~cov_last[qi_, qj_] & ~sky_wan[qi_, qj_] & nobj_, "sky": sky_wan[qi_, qj_] & nobj_}
            per, size = summarise(tr, vis, q0, obj_wan, px, T_v, sky_wan=sky_wan, region_wan=region, F_expected=F_exp, source=src_tracks, F_obj_expected=F_obj, outside_split=split,
                                  source_dyn=src_dyn, nonrigid_region=region_mask, ground_wan=ground_wan, coverage_split=cov_split)
            if str(vp) == "source":
                src_tracks = (tr, vis)
                if Msrc is not None:
                    kd_ = np.ones((2 * args.dyn_dilate_px + 1, 2 * args.dyn_dilate_px + 1), np.uint8)
                    src_dyn = np.zeros(tr.shape[:2], bool)
                    for f_ in range(tr.shape[0]):
                        mf = cv2.dilate(cv2.resize(Msrc[t_a + f_].astype(np.uint8), (WAN_W, WAN_H), interpolation=cv2.INTER_NEAREST), kd_).astype(bool)
                        src_dyn[f_] = mf[np.clip(tr[f_, :, 1].astype(int), 0, WAN_H - 1), np.clip(tr[f_, :, 0].astype(int), 0, WAN_W - 1)]
                    print(f"[grader] source dynamic mask: {src_dyn.mean():.3f} of source track-frames excluded from the direction reference", flush=True)
            # the tile detector (reviewer 2026-10-10): background = outside the frame-0 object mask (dilated) and outside the sky; every 4th frame
            bg_m = ~cv2.dilate(obj_wan.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool) & ~sky_wan
            tiles = [GD.tile_score(video[4 * j], bg_m) for j in range(T_v)]
            tile_vals = [x["secondary_autocorr_peak"] for x in tiles if x["secondary_autocorr_peak"] is not None]
            tile = {"median": float(np.median(tile_vals)) if tile_vals else None, "max": float(np.max(tile_vals)) if tile_vals else None,
                    "lag_px_at_max": tiles[int(np.argmax(tile_vals))]["lag_px"] if tile_vals else None, "per_frame": tiles}
            last = per[-1]; mid = per[T_v // 2]
            rows.append({"pair": pair, "video": vp.name, "beta_tag": beta_tag, "expectation_bridge": vtag, "continuity": continuity, "render_continuity": render_cont, "T_l": T_v,
                         "ground_mask": args.ground_mask, "expectation_note": "every expected column is the TARGET geometry (bridged camera on the anchor lift, c_tar on the lattice); on the source row it is not the source's own expectation", "object_size_patch": size, "per_frame": per, "tiles": tile,
                         "expected_bg_disp_median_patch": exp_bg[:T_v], "expected_obj_centroid_disp_patch": exp_obj[:T_v],
                         "expected_bg_in_region": exp_in[:T_v] if exp_in else None, "expected_bg_in_region_dense_grid": exp_in_dense[:T_v] if exp_in_dense else None,
                         "summary": {"bg_last_measured_vs_expected": [last["bg_disp_median_patch"], exp_bg[T_v - 1]], "bg_mid_measured_vs_expected": [mid["bg_disp_median_patch"], exp_bg[T_v // 2]],
                                     "bg_in_region_last_measured_vs_expected": [last.get("bg_in_region_median_patch"), exp_in[T_v - 1] if exp_in else None],
                                     "bg_in_region_last_expected_dense_grid": exp_in_dense[T_v - 1] if exp_in_dense else None,
                                     "bg_in_region_last_expected_at_visible": last.get("bg_in_region_expected_at_visible_patch"),
                                     "bg_out_region_last": last.get("bg_out_region_median_patch"), "n_bg_in_region_last": last.get("n_bg_in_region"),
                                     "outside_cos_vs_source_last": last.get("outside_cos_vs_source_median"), "n_outside_cos_last": last.get("n_outside_cos"),
                                     "outside_parts_last": last.get("outside_parts"),
                                     "obj_centroid_expected_at_survivors_last": last.get("obj_centroid_expected_at_survivors_norm"),
                                     "obj_similarity_last": {"measured": last.get("obj_similarity_measured"), "render": last.get("obj_similarity_render")},
                                     "obj_centroid_gait_oscillation_patch": last.get("obj_centroid_gait_oscillation_patch"),
                                     "obj_centroid_gait_oscillation_rel_patch": last.get("obj_centroid_gait_oscillation_rel_patch"), "n_local_bg_tracks": last.get("n_local_bg_tracks"),
                                     "obj_rel_displacement_last_patch": last.get("obj_rel_displacement_last_patch"), "obj_rel_path_length_patch": last.get("obj_rel_path_length_patch"),
                                     "nonrigid_region_resid_last": [last.get("nonrigid_region_resid_median_patch"), last.get("nonrigid_region_resid_p90_patch"), last.get("n_nonrigid_region")],
                                     "nonrigid_region_resid_mid": [mid.get("nonrigid_region_resid_median_patch"), mid.get("nonrigid_region_resid_p90_patch")],
                                     "bg_in_region_ratio_to_own_expectation": (last.get("bg_in_region_median_patch") / last["bg_in_region_expected_at_visible_patch"]
                                                                               if last.get("bg_in_region_median_patch") is not None and last.get("bg_in_region_expected_at_visible_patch") else None),
                                     "obj_last_measured_vs_expected": [last.get("obj_centroid_disp_norm"), exp_obj[T_v - 1]],
                                     "obj_deformation_last": last.get("obj_deformation"), "obj_deformation_max": max((p.get("obj_deformation") or 0) for p in per),
                                     "obj_scale_change_last": last.get("obj_scale_change"), "obj_track_survival_last": last["obj_track_survival"],
                                     "obj_track_survival_min": min(p["obj_track_survival"] for p in per),
                                     "tile_peak_median": tile["median"], "tile_peak_max": tile["max"], "continuity_first_over_median": continuity["first_over_median"],
                                     "continuity_first_step": continuity["first_step"], "continuity_median_step": continuity["median_step"],
                                     "render_continuity_first_over_median": render_cont["first_over_median"] if render_cont else None,
                                     "render_continuity_first_step": render_cont["first_step"] if render_cont else None,
                                     "track_classes_last": last.get("track_classes"), "local_ground": last.get("local_ground")}})
            print(f"[A0] {pair} {vp.name}: bg last {last['bg_disp_median_patch']} vs expected {exp_bg[-1]:.2f}; obj centroid {last.get('obj_centroid_disp_norm')} vs {exp_obj[-1]:.2f}; deformation max {rows[-1]['summary']['obj_deformation_max']:.3f}")
    rep = root / "reports" / "m3_s4"; rep.mkdir(parents=True, exist_ok=True)
    jf = rep / "a0_motion.json"
    prev = json.loads(jf.read_text(encoding="utf-8")) if jf.exists() else []
    key = lambda r: (r["pair"], r["video"])                                                          # noqa: E731
    dup = {key(r) for r in prev} & {key(r) for r in rows}
    assert not dup or args.force, f"rule 24: {sorted(dup)} already recorded; pass --force"
    rows = sorted([r for r in prev if key(r) not in {key(x) for x in rows}] + rows, key=key)
    jf.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    f = lambda v, n=2: "-" if v is None else f"{v:.{n}f}"                                           # noqa: E731
    md = ["# Module 3 — unguided A0 videos: camera and object motion measured (CoTracker on the generated video; the same instrument grades S4)", "",
          "| pair | video | object size (patch) | background displacement, mid / last: measured vs expected (patch) | bg INSIDE the render mask, last: measured vs expected over this video's visible tracks [full grid] (n) = ratio / OUTSIDE [between paired A0 and source?] / outside direction: median cos vs source (n) / outside parts uncovered-at-0, covered-but-excluded: median (cos, n) | object centroid, last: measured vs expected at the SAME surviving tracks [whole object] / similarity scale measured vs render / gait oscillation of the centroid, absolute / relative to the local background (peak, patch; SG 13) / locomotion: displacement vs local background, last (patch) | non-rigid residual in --region-mask, mid / last: median (p90) patch (n) | continuity: first frame step ÷ median step (> 2 = the given frame was not continued) | object centroid displacement, last: measured vs expected | object deformation last / max (0 = rigid) | object scale change last | object track survival last / min (1 = all frame-0 object tracks still visible) | tile detector: secondary autocorrelation peak, median / max over frames |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    by_name = {(r["pair"], r["video"]): r for r in rows}

    def parts_str(p):
        if not p:
            return "-"
        return ", ".join(f"{k[:9]} {f(v.get('median_patch'))} ({f(v.get('cos_vs_source_median'))}, {v.get('n_cos') or v.get('n')})" for k, v in p.items() if k != "sky")

    def nr_str(v):
        if not v or v[0] is None:
            return "-"
        return f"{v[0]:.2f} ({v[1]:.2f})" + (f" ({v[2]})" if len(v) > 2 and v[2] is not None else "")

    def sim_str(sm):
        if not sm or not sm.get("measured") or not sm.get("render"):
            return "-"
        return f"{sm['measured']['scale']:.2f} vs {sm['render']['scale']:.2f}"

    def outside_bracket(r):
        """For a guided video A?_seed<k>_<bridge>_F<n>...: is its outside-mask magnitude between the paired A0's and the source's (either order)?"""
        m_ = re.match(r"A[23]_seed(\d+)_(\w+?)_F(\d+)", r["video"])
        if not m_:
            return None
        a0 = by_name.get((r["pair"], f"A0_seed{m_.group(1)}_rec_F{m_.group(3)}.mp4")); src = by_name.get((r["pair"], "source"))
        v = r["summary"].get("bg_out_region_last")
        if a0 is None or src is None or v is None:
            return None
        lo, hi = sorted([a0["summary"].get("bg_out_region_last") or 0.0, src["summary"].get("bg_out_region_last") or 0.0])
        return "yes" if lo - 1e-9 <= v <= hi + 1e-9 else f"no ({lo:.2f}-{hi:.2f})"
    for r in rows:
        s = r["summary"]
        ir = s.get("bg_in_region_last_measured_vs_expected") or [None, None]
        md.append(f"| {r['pair'][-22:]} | {r['video']} | {f(r['object_size_patch'], 1)} | {f(s['bg_mid_measured_vs_expected'][0])} vs {f(s['bg_mid_measured_vs_expected'][1])} / {f(s['bg_last_measured_vs_expected'][0])} vs {f(s['bg_last_measured_vs_expected'][1])} | "
                  f"{f(ir[0])} vs {f(s.get('bg_in_region_last_expected_at_visible'))} [{f(s.get('bg_in_region_last_expected_dense_grid'))}] ({s.get('n_bg_in_region_last')}) = {f(s.get('bg_in_region_ratio_to_own_expectation'))} / {f(s.get('bg_out_region_last'))} [{outside_bracket(r) or '-'}] / {f(s.get('outside_cos_vs_source_last'))} ({s.get('n_outside_cos_last') or '-'}) / {parts_str(s.get('outside_parts_last'))} | "
                  f"{f((s.get('obj_last_measured_vs_expected') or [None])[0])} vs {f(s.get('obj_centroid_expected_at_survivors_last'))} [{f((s.get('obj_last_measured_vs_expected') or [None, None])[1])}] / {sim_str(s.get('obj_similarity_last'))} / {f(s.get('obj_centroid_gait_oscillation_patch'))} / {f(s.get('obj_centroid_gait_oscillation_rel_patch'))} / {f(s.get('obj_rel_displacement_last_patch'))} | {nr_str(s.get('nonrigid_region_resid_mid'))} / {nr_str(s.get('nonrigid_region_resid_last'))} | {f(s.get('continuity_first_over_median'), 1)} | "
                  f"{f(s['obj_last_measured_vs_expected'][0])} vs {f(s['obj_last_measured_vs_expected'][1])} | {f(s['obj_deformation_last'], 3)} / {f(s['obj_deformation_max'], 3)} | {f(s['obj_scale_change_last'], 3)} | "
                  f"{f(s.get('obj_track_survival_last'))} / {f(s.get('obj_track_survival_min'))} | {f(s.get('tile_peak_median'), 3)} / {f(s.get('tile_peak_max'), 3)} |")
    md += ["", "## Track classes by coverage, continuity in absolute values, and the ground reference (reviewer 2026-10-14)", "",
           "| pair | video | tracks at the last frame: covered / uncovered / sky / object (in mask / out of mask) | video continuity: first step / median step (ratio) | render continuity: first / median (ratio) | local ground |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        s_ = r["summary"]; tc = s_.get("track_classes_last") or {}
        md.append(f"| {r['pair']} | {r['video']} | {tc.get('covered', '-')} / {tc.get('uncovered', '-')} / {tc.get('sky', '-')} / {tc.get('object', '-')} ({tc.get('in_mask', '-')} / {tc.get('out_of_mask', '-')}); "
                  f"cos vs source covered {f(tc.get('covered_cos_vs_source'))} ({tc.get('covered_n_cos', '-')}) uncovered {f(tc.get('uncovered_cos_vs_source'))} ({tc.get('uncovered_n_cos', '-')}) | "
                  f"{f(s_.get('continuity_first_step'))} / {f(s_.get('continuity_median_step'))} ({f(s_.get('continuity_first_over_median'))}) | "
                  f"{f(s_.get('render_continuity_first_step'))} / {f(s_.get('render_continuity_first_over_median') and r.get('render_continuity', {}).get('median_step'))} ({f(s_.get('render_continuity_first_over_median'))}) | {s_.get('local_ground') or '-'} |")
    md += ["", "Every 'expected' column above is the TARGET's geometry (the bridged camera on the anchor's lift; c_tar on the lattice). The source row is read against the same "
           "expectation, so it is not a self-test of the source: the source's own agreement with that expectation at the same pixels is the pixel-bridge record "
           "(scripts/check_m3_pixel_bridge.py), and the source's object screen motion is the mask-centroid arbiter (scripts/check_m3_mask_centroid.py)."]
    md += ["", "Outside-mask criterion (reviewer 2026-10-11): a guided run's outside-mask magnitude lies between the paired A0's and the source's, and the median cosine between its "
           "outside tracks and the source's outside tracks (same grid, both visible, source displacement >= 0.25 patch) is >= 0.8; grade 'source' in the same invocation."]
    md += ["", "Paired criteria (reviewer 2026-10-10): a guided run is read against the A0 of the SAME seed and length: its object track survival must not be below "
           "the paired A0's (an A0 that loses every track says nothing about guidance); its tile peak must lie within the A0-to-source range (grade 'source' alongside); "
           "its background displacement must move more than 0.5 patch from the A0 value toward the expected one."]
    md += ["", "Reading: background displacement is the camera the model chose (expected = the source camera through the bridge of record on the lifted background); the object "
           "centroid is where it moved the object (expected = the retargeted path); deformation is the median spread of object tracks about their centroid in object sizes, "
           "which separates 'the model moved the object' from 'the model distorted it'. A guided run is graded with the same columns; the claim rule applies."]
    (rep / "a0_motion.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
