#!/usr/bin/env python
"""CoTracker3 probe (CLAUDE.md golden rule 1: probe before adapter). D-M3-05 (John, 2026-10-07): CoTracker is the tracker for
the attention-readout ground truth. Measures its conventions on one real clip at Wan's frame size; writes
<drive_root>/reports/probe_cotracker/probe_cotracker.{md,json}. The readout adapter asserts what this measures.

  C1 call contract     torch.hub entry point, input dtype/range/shape, query format (t, x, y), outputs and shapes
  C2 pixel convention  x = column, y = row? pixel centre at 0 or 0.5? A query at a known pixel must return itself at its own frame
  C3 visibility        dtype/range of the visibility output; fraction visible over 81 frames at 832x480 on a grid of patch centres
  C4 static sanity     on frames where the camera is nearly still, grid points must move < 1 px (which clip: 370)
  C5 cost              wall time, peak VRAM, for 1560 queries over 81 frames

    pip install -q -U imageio-ffmpeg   (CoTracker comes from torch.hub: facebookresearch/co-tracker, cotracker3_offline)
    python scripts/probe_cotracker.py --clip 000000000370.1.008
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WAN_W, WAN_H, PX = 832, 480, 16


def read_video(path: Path, n: int, W: int, H: int) -> np.ndarray:
    import cv2
    assert Path(path).exists(), f"{path} missing: run scripts/bootstrap_phase2.sh (it downloads the reference clips to local disk)"
    cap = cv2.VideoCapture(str(path)); fr = []
    while len(fr) < n:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)[..., ::-1])
    cap.release()
    assert len(fr) == n, f"{path}: {len(fr)} frames read, {n} needed"
    return np.stack(fr)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", default=__import__("os").environ.get("MT_ROOT", "/content/drive/MyDrive/motion_transfer"))
    ap.add_argument("--videos", default=__import__("os").environ.get("MT_DATA", "/content/data") + "/videos")
    ap.add_argument("--clip", default="000000000370.1.008")
    ap.add_argument("--frames", type=int, default=81)
    args = ap.parse_args(argv)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(dev).eval()
    video = read_video(Path(args.videos) / f"{args.clip}.mp4", args.frames, WAN_W, WAN_H)        # [T,H,W,3] uint8 RGB
    vt = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().to(dev)                            # [1,T,3,H,W] 0..255 (documented contract: tested below)
    jj, ii = np.meshgrid(np.arange(WAN_W // PX) + 0.5, np.arange(WAN_H // PX) + 0.5)
    grid = np.stack([jj.ravel() * PX, ii.ravel() * PX], 1)                                           # patch centres in pixels (x, y)
    queries = torch.from_numpy(np.concatenate([np.zeros((len(grid), 1)), grid], 1)).float().to(dev)[None]   # [1,N,3] = (t, x, y)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        tracks, vis = model(vt, queries=queries)
    wall = time.time() - t0
    tr, vs = tracks[0].float().cpu().numpy(), vis[0].float().cpu().numpy()
    rec = {"clip": args.clip, "entry": "facebookresearch/co-tracker:cotracker3_offline", "device": dev,
           "C1": {"input": "video [1,T,3,H,W] float 0..255; queries [1,N,3] = (frame, x, y)", "tracks_shape": list(tr.shape), "vis_shape": list(vs.shape),
                  "tracks_dtype": str(tracks.dtype), "vis_dtype": str(vis.dtype), "vis_min_max": [float(vs.min()), float(vs.max())]}}
    # C2: a query at its own frame must return itself (pixel convention); and column/row order
    d0 = np.abs(tr[0] - grid)
    rec["C2"] = {"self_return_max_abs_px": float(d0.max()), "note": "if ~0: tracks are (x, y) in the same pixel coordinates as the queries"}
    # C3: visibility
    rec["C3"] = {"visible_frac_per_frame_every8": [round(float(v), 3) for v in (vs > 0.5).mean(1)[::8]], "overall": float((vs > 0.5).mean())}
    # C4: static sanity between consecutive frames (median step of all points)
    step = np.linalg.norm(np.diff(tr, axis=0), axis=-1)
    rec["C4"] = {"median_step_px_per_frame": float(np.median(step)), "p90_step_px": float(np.quantile(step, 0.9)), "max_total_drift_px": float(np.linalg.norm(tr[-1] - tr[0], axis=-1).max())}
    # C6 (reviewer 2026-10-07d): CoTracker against Stage A's 2-D tracks on STATIC background points — an independent tracker on the
    # same points, same frames. Agreement in lattice patches is the error bar the readout gate inherits.
    c6 = None
    try:
        root = Path(args.drive_root)
        with np.load(root / "cache" / "stage_a" / args.clip / "bundle.npz") as z:
            Pa, Qa = z["P"].astype(np.float64), z["Q"].astype(bool)
        with np.load(root / "cache" / "stage_b" / f"{args.clip}.npz") as z:
            y_dyn = z["y_dyn"].astype(bool)
        from src.preprocess.stage_c import finalize
        M, _, _ = finalize.read_mask(root / "cache" / "stage_c" / args.clip)
        F = min(args.frames, Qa.shape[1])
        inm = np.zeros(len(Pa), bool)
        for t in range(F):
            u_, v_ = np.clip(np.rint(Pa[:, t, 0]).astype(int), 0, 255), np.clip(np.rint(Pa[:, t, 1]).astype(int), 0, 255)
            inm |= Qa[:, t] & M[t, v_, u_]
        static = ~y_dyn & ~inm & Qa[:, 0] & (Qa[:, :F].mean(1) > 0.9)
        idx = np.nonzero(static)[0]
        rng = np.random.default_rng(0); idx = rng.choice(idx, min(600, len(idx)), replace=False)
        sx, sy = WAN_W / 256.0, WAN_H / 256.0
        q2 = torch.from_numpy(np.stack([np.zeros(len(idx)), Pa[idx, 0, 0] * sx, Pa[idx, 0, 1] * sy], 1)).float().to(dev)[None]
        with torch.no_grad():
            tr2, vs2 = model(vt, queries=q2)
        tr2, vs2 = tr2[0].float().cpu().numpy(), vs2[0].float().cpu().numpy()
        d_cot = tr2[:F] - tr2[0]                                                    # displacement from frame 0, Wan px
        d_a = np.stack([Pa[idx, t, 0] * sx - Pa[idx, 0, 0] * sx for t in range(F)], 0)
        d_a = np.stack([d_a, np.stack([Pa[idx, t, 1] * sy - Pa[idx, 0, 1] * sy for t in range(F)], 0)], -1)
        ok = Qa[idx][:, :F].T & (vs2[:F] > 0.5)
        gap = np.linalg.norm(d_cot - d_a, axis=-1)[ok] / PX
        c6 = {"n_static_tracks": int(len(idx)), "n_obs": int(ok.sum()), "gap_patches_median": float(np.median(gap)), "gap_patches_p90": float(np.quantile(gap, 0.9)),
              "frac_within_quarter_patch": float((gap < 0.25).mean()), "frac_within_half_patch": float((gap < 0.5).mean()),
              "stage_a_motion_patches_median": float(np.median(np.linalg.norm(d_a, axis=-1)[ok]) / PX)}
    except Exception as e:
        c6 = {"error": f"{type(e).__name__}: {e}"}
    rec["C6"] = c6
    rec["C5"] = {"wall_s": round(wall, 1), "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if dev == "cuda" else None, "n_queries": int(len(grid)), "n_frames": args.frames}
    out = Path(args.drive_root) / "reports" / "probe_cotracker"; out.mkdir(parents=True, exist_ok=True)
    (out / "probe_cotracker.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    md = [f"# CoTracker3 probe on {args.clip} ({args.frames} frames at {WAN_W}x{WAN_H}, {dev})", "",
          f"- C1 contract: tracks {rec['C1']['tracks_shape']} {rec['C1']['tracks_dtype']}, visibility {rec['C1']['vis_shape']} {rec['C1']['vis_dtype']} in {rec['C1']['vis_min_max']}",
          f"- C2 pixel convention: a query returns itself at its own frame to {rec['C2']['self_return_max_abs_px']:.3f} px",
          f"- C3 visibility: overall {rec['C3']['overall']:.3f}; per frame (every 8): {rec['C3']['visible_frac_per_frame_every8']}",
          f"- C4 motion: median step {rec['C4']['median_step_px_per_frame']:.2f} px/frame, p90 {rec['C4']['p90_step_px']:.2f}, max total drift {rec['C4']['max_total_drift_px']:.1f} px",
          f"- C5 cost: {rec['C5']['wall_s']} s, {rec['C5']['peak_vram_gb']} GB",
          f"- C6 vs Stage-A static 2-D tracks (independent tracker, same points): {json.dumps(rec['C6'])}", "",
          "Decisions for the human: confirm C1/C2 as the conventions the readout adapter asserts; whether visibility > 0.5 is the mask for ground-truth flow."]
    (out / "probe_cotracker.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
