#!/usr/bin/env python
"""Export one clip's COMPUTED rigid-motion track as a standalone .npz.

The raw record already exists in `cache/stage_b/<clip>.npz` as the `fc_*`
fields (mixed in with the rest of Stage B). This extracts just the
rigid-motion arrays, recovers the pivot READ-ONLY (plain support centroids -
no re-fitting), VERIFIES that it reproduces the stored tau_rel, and writes a
documented npz + manifest.

    python scripts/export_rigid_motion_npz.py --clip-id 000000000035.1.003

Semantics (CLAUDE.md rule 9, centroid-anchored): for each frame t with a
valid fit, the blob's motion from the seed frame is
    X_cam_t ~= R_rel[t] @ (X_cam_seed - c_ref[t]) + c_ref[t] + tau_rel[t]
with c_ref[t] = centroid of that frame's support AT THE SEED FRAME and
tau_rel[t] = c_t[t] - c_ref[t]. World-origin tau is never stored.

PROVISIONAL: the blob includes the bleed halo (object-vs-halo is not
geometrically separable); the contract re-fits this downstream on
mask-refined dense tracks. RELATIVE, not world: composing world motion
(camera trajectory o relative motion) and its trajectory source are open
decisions owned by the retargeting SKILL - not done here. `G` is included so
a figure can show the camera track, NOT as a composed world motion.

Writes <drive_root>/viz/thesis/<clip_id>/rigid_motion.npz + rigid_motion_npz.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess.bundle import read_bundle                 # noqa: E402
from src.preprocess.config import load_config                 # noqa: E402
from scripts.plot_rigid_motion import angle_axis, read_events  # noqa: E402

PIVOT_TOL = 1e-4      # model units; f32 vs f64 centroid rounding is ~1e-6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-id", default="000000000035.1.003")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config("configs/stage_a.yaml")
    c = args.clip_id
    bdir = Path(cfg["paths"]["stage_b_cache"])
    with np.load(bdir / f"{c}.npz") as z:
        sb = {k: z[k] for k in z.files}
    qb = json.loads((bdir / f"{c}_qc_b.json").read_text(encoding="utf-8"))
    fc = qb.get("follow_cam") or {}
    channel = "branch" if fc.get("branch_active") else "x0"
    if "fc_R_rel" not in sb:
        raise SystemExit(
            f"{c}: no fc_* record (channel = {channel}). The x0-channel "
            "world-route rigid fit is NOT implemented (retargeting module, not "
            "started) - there is no computed rigid track to export; see "
            "docs/contract_o_pre_handoff.md sec 6.")
    b = read_bundle(cfg, c)
    Xl, Q, G = b["X_local"], b["Q"], b["G"]
    R, tau = sb["fc_R_rel"], sb["fc_tau_rel"]
    ok = sb["fc_rel_valid"].astype(bool)
    blob = sb["fc_blob_idx"].astype(int)
    F = len(R)
    ref = fc.get("seed_frame")
    assert ref is not None, "qc_b.follow_cam.seed_frame missing - cannot anchor tau"
    ref = int(ref)
    ang, axis = angle_axis(R)

    # Pivot recovery: exactly the centroids the fit used (follow_cam._kabsch
    # muA/muB over the per-frame common support). Arithmetic, not a re-fit.
    c_ref = np.full((F, 3), np.nan)
    c_t = np.full((F, 3), np.nan)
    n_sup = np.zeros(F, np.int32)
    for t in range(F):
        common = blob[Q[blob, t] & Q[blob, ref]]
        n_sup[t] = len(common)
        if not len(common):
            continue
        c_ref[t] = Xl[common, ref].astype(np.float64).mean(0)
        c_t[t] = Xl[common, t].astype(np.float64).mean(0)
    err = np.linalg.norm((c_t - c_ref) - tau, axis=1)
    worst = float(np.nanmax(err[ok])) if ok.any() else float("nan")
    pivot_ok = np.isfinite(worst) and worst <= PIVOT_TOL

    o = {"clip_id": np.array(c), "channel": np.array(channel),
         "ref_frame": np.array(ref, np.int32),
         "R_rel": R.astype(np.float32), "tau_rel": tau.astype(np.float32),
         "rel_valid": ok, "angle_deg": ang.astype(np.float32),
         "axis": axis.astype(np.float32), "n_support": n_sup,
         "blob_idx": blob.astype(np.int32),
         "blob_core_idx": sb.get("fc_blob_core_idx",
                                 blob.astype(np.int32)).astype(np.int32),
         "G": G.astype(np.float32)}
    if pivot_ok:
        o["c_ref"] = c_ref.astype(np.float32)
        o["c_t"] = c_t.astype(np.float32)
    # critical frames for marking the horizontal axis of a figure (measured;
    # absent sources are omitted, never placeholder-filled)
    ev = read_events(cfg, c, F)
    if ev["boundaries"]:
        o["window_boundaries"] = np.asarray(ev["boundaries"], np.int32)
    if ev["t_c"] is not None:
        o["t_c"] = np.array(ev["t_c"], np.int32)
    if ev["present"] is not None:
        o["object_present"] = ev["present"]
    doc = {
        "clip_id": "clip id", "channel": "Stage B channel that owns the clip",
        "ref_frame": "seed frame the fit is relative to",
        "R_rel": "[F,3,3] rotation, camera frame, vs the seed frame (NaN where invalid)",
        "tau_rel": "[F,3] centroid displacement c_t - c_ref (NaN where invalid)",
        "rel_valid": "[F] frames with a valid fit - mask everything by this",
        "angle_deg": "[F] rotation angle of R_rel, degrees (derived here)",
        "axis": "[F,3] unit rotation axis (derived here; undefined at 0/180 deg)",
        "n_support": "[F] tracks in the frame's common support",
        "blob_idx": "quiet-blob track indices (subject + halo) into the Stage A arrays",
        "blob_core_idx": "centroid-proximal core subset of blob_idx",
        "G": "[F,4,4] camera pose G_{0<-t} (camera track; NOT a composed world motion)",
        "c_ref": "[F,3] pivot: support centroid AT THE SEED FRAME (recovered, verified)",
        "c_t": "[F,3] support centroid at frame t (recovered, verified)",
        "window_boundaries": "encode window boundaries - a step HERE is a re-gauge "
                             "artefact suspect, not object motion",
        "t_c": "Stage C canonical / prompt frame",
        "object_present": "[F] Stage C honest presence flag (false = not there)"}
    root = Path(str(cfg["paths"]["drive_root"]))
    out = Path(args.out) if args.out else root / "viz" / "thesis" / c
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "rigid_motion.npz", **o)

    nrm = np.linalg.norm(tau, axis=1)
    md = [f"# rigid_motion.npz - {c}", "",
          "| key | shape | dtype | meaning |", "|---|---|---|---|"]
    for k, v in o.items():
        a = np.asarray(v)
        md.append(f"| `{k}` | `{'x'.join(str(s) for s in a.shape) or 'scalar'}` "
                  f"| {a.dtype} | {doc.get(k, '')} |")
    md += ["",
           f"- raw source: `cache/stage_b/{c}.npz` fields `fc_R_rel`, `fc_tau_rel`, "
           f"`fc_rel_valid`, `fc_blob_idx`, `fc_blob_core_idx`; seed frame from "
           f"`{c}_qc_b.json` follow_cam.seed_frame; `G` from the Stage A bundle",
           f"- channel = **{channel}**; valid fits {int(ok.sum())}/{F} frames; "
           f"rotation angle median {np.nanmedian(ang):.2f} deg, max "
           f"{np.nanmax(ang):.2f} deg; ||tau_rel|| median {np.nanmedian(nrm):.4f}, "
           f"max {np.nanmax(nrm):.4f} (D4RT model units)",
           "- reconstruction: `X_cam_t ~= R_rel[t] @ (X_cam_seed - c_ref[t]) + "
           "c_ref[t] + tau_rel[t]` (centroid-anchored, CLAUDE.md rule 9)",
           f"- critical frames for a figure axis: seed = {ref}, t_c = "
           f"{o['t_c'] if 't_c' in o else 'n/a'}, encode window boundaries = "
           f"{[int(x) for x in o['window_boundaries']] if 'window_boundaries' in o else 'n/a'}; "
           f"invalid-fit frames = {np.flatnonzero(~ok).tolist()}. A step AT a "
           "window boundary is a re-gauge artefact suspect, not object motion",
           (f"- pivot recovered read-only from `X_local` + `blob_idx` and VERIFIED: "
            f"max |(c_t - c_ref) - tau_rel| = {worst:.2e} over valid frames "
            f"(tolerance {PIVOT_TOL:.0e}) - `c_ref`/`c_t` included"
            if pivot_ok else
            f"- pivot NOT included: recovery check FAILED, max |(c_t - c_ref) - "
            f"tau_rel| = {worst:.2e} > {PIVOT_TOL:.0e}. Stage B does not persist "
            f"the pivot (`c_rel` holds the blob compactness centre and is never "
            f"written), so the support used by the fit could not be reproduced - "
            f"a FINDING to report, not something to patch downstream"),
           "- PROVISIONAL: blob includes the bleed halo, pre-SAM2; the contract "
           "re-fits downstream on mask-refined dense tracks",
           "- RELATIVE, not world motion: on branch clips world motion is composed "
           "(camera trajectory o relative motion) and the trajectory source (bundle "
           "`G` vs the A.5 ViPE record) is an OPEN DECISION - do not present a "
           "composed world track as computed"]
    (out / "rigid_motion_npz.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\n-> {out / 'rigid_motion.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
