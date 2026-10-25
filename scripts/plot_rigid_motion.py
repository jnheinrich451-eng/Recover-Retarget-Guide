#!/usr/bin/env python
"""Thesis figure for the PROVISIONAL rigid motion of one clip's subject,
read straight from the Stage B cache. Read-only; nothing is re-fitted.

    python scripts/plot_rigid_motion.py --clip-id 000000000035.1.003

What this plots (branch/follow-cam clips only): fc_R_rel / fc_tau_rel, the
per-frame Kabsch fit of the quiet blob against the seed frame, in the CAMERA
frame, centroid-anchored (CLAUDE.md rule 9: tau = centroid displacement).

Critical frames marked on the horizontal axis (all measured, none invented):
  seed      the reference frame the fit is relative to (qc_b follow_cam)
  t_c       Stage C canonical / prompt frame
  w<i>      encode window boundaries (bundle qc.window_boundaries) - a step
            here is a re-gauge artefact suspect, NOT an object-motion finding
            (cross-window rigid fits need segment_scales de-gauging)
  shaded    frames with no valid fit; hatched: Stage C object_present = false

Honest scope, stated on the figure:
  - provisional: the blob includes the bleed halo (object-vs-halo is not
    geometrically separable); the contract re-fits it downstream on
    mask-refined dense tracks.
  - relative, NOT world motion: on branch clips world motion is COMPOSED
    (camera trajectory o relative motion) and the trajectory source is an
    open decision owned by the retargeting SKILL - not plotted here.
  - x0-channel clips have no fc_* record: their world-route rigid fit is not
    implemented yet, so this script refuses them instead of inventing one.

Writes <drive_root>/viz/thesis/<clip_id>/fig_rigid_motion.{pdf,png} + .md
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
from src.preprocess.stage_c import finalize                   # noqa: E402
from scripts.plot_thesis_tracks import (INK, INK2, MASK, MUTED, SUBJ, DENSE,  # noqa: E402
                                        save, style)


def angle_axis(R):
    """[F,3,3] -> angle [F] in deg and unit axis [F,3]; NaN where R is NaN."""
    F = len(R)
    ang = np.full(F, np.nan)
    axis = np.full((F, 3), np.nan)
    for t in range(F):
        M = R[t].astype(np.float64)
        if not np.isfinite(M).all():
            continue
        c = (np.trace(M) - 1.0) / 2.0
        th = float(np.arccos(np.clip(c, -1.0, 1.0)))
        ang[t] = np.degrees(th)
        s = np.sin(th)
        if s > 1e-8:                       # axis undefined at th = 0 / pi
            axis[t] = np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0],
                                M[1, 0] - M[0, 1]]) / (2 * s)
    return ang, axis


def read_events(cfg, clip_id, F):
    """Measured critical frames. Absent sources are omitted, never faked."""
    ev = {"boundaries": [], "t_c": None, "present": None}
    try:
        b = read_bundle(cfg, clip_id)
        ev["boundaries"] = [int(x) for x in b["qc"].get("window_boundaries", [])
                            if 0 < int(x) < F]
    except Exception as e:                 # diagnostics only - never block a figure
        print(f"[events] bundle qc unavailable ({e.__class__.__name__}): "
              "window boundaries not marked")
    croot = cfg["paths"].get("stage_c_cache")
    cdir = Path(croot) / clip_id if croot else None
    if cdir is not None and (cdir / "qc_c.json").exists():
        _, present, t_c = finalize.read_mask(cdir)
        ev["t_c"] = int(t_c)
        ev["present"] = np.asarray(present, bool)
    return ev


def spans(mask):
    """Contiguous True runs of a bool array -> [(t0, t1), ...] inclusive."""
    out, run = [], None
    for t, v in enumerate(mask):
        if v and run is None:
            run = t
        elif not v and run is not None:
            out.append((run, t - 1)); run = None
    if run is not None:
        out.append((run, len(mask) - 1))
    return out


def mark_axis(ax, ev, ref, F, labels=True):
    """Critical frames ON the horizontal axis: shaded spans, rules, top ticks."""
    for t0, t1 in ev["_invalid"]:
        ax.axvspan(t0 - 0.5, t1 + 0.5, color=MUTED, alpha=0.18, lw=0, zorder=0)
    for t0, t1 in ev["_absent"]:
        ax.axvspan(t0 - 0.5, t1 + 0.5, facecolor="none", edgecolor=MASK,
                   hatch="///", lw=0, alpha=0.5, zorder=0)
    for t in ev["boundaries"]:
        ax.axvline(t, color=DENSE, lw=0.9, ls="--", alpha=0.8, zorder=1)
    marks = {}
    if ref is not None:
        marks.setdefault(int(ref), []).append("seed")
        ax.axvline(int(ref), color=MUTED, lw=1, ls="-.", zorder=1)
    if ev["t_c"] is not None:
        marks.setdefault(int(ev["t_c"]), []).append("t_c")
        ax.axvline(int(ev["t_c"]), color=INK2, lw=0.9, ls=":", zorder=1)
    for i, t in enumerate(ev["boundaries"], 1):
        marks.setdefault(int(t), []).append(f"w{i}")
    if labels and marks:
        sec = ax.secondary_xaxis("top")
        sec.set_xticks(sorted(marks))
        sec.set_xticklabels(["/".join(marks[t]) for t in sorted(marks)],
                            fontsize=6, color=INK2)
        sec.tick_params(length=2.5, colors=MUTED)
        for s in sec.spines.values():
            s.set_visible(False)
    return sorted(marks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-id", default="000000000035.1.003")
    ap.add_argument("--no-events", action="store_true",
                    help="plain frame axis, no critical-frame marks")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config("configs/stage_a.yaml")
    c = args.clip_id
    bdir = Path(cfg["paths"]["stage_b_cache"])
    with np.load(bdir / f"{c}.npz") as z:
        arr = {k: z[k] for k in z.files}
    qb = json.loads((bdir / f"{c}_qc_b.json").read_text(encoding="utf-8"))
    fc = qb.get("follow_cam") or {}
    if "fc_R_rel" not in arr:
        raise SystemExit(
            f"{c}: no fc_* record (channel = "
            f"{'branch' if fc.get('branch_active') else 'x0'}). The x0-channel "
            "world-route rigid fit is NOT implemented (retargeting module, not "
            "started) - nothing to plot; see docs/contract_o_pre_handoff.md sec 6.")
    R, tau, ok = arr["fc_R_rel"], arr["fc_tau_rel"], arr["fc_rel_valid"].astype(bool)
    F = len(R)
    ref = fc.get("seed_frame")
    ang, axis = angle_axis(R)
    mag = np.linalg.norm(tau, axis=1)
    for a in (ang, mag):
        a[~ok] = np.nan
    axis[~ok] = np.nan
    tau = np.where(ok[:, None], tau, np.nan)

    ev = {"boundaries": [], "t_c": None, "present": None} if args.no_events \
        else read_events(cfg, c, F)
    ev["_invalid"] = spans(~ok)
    ev["_absent"] = ([] if ev["present"] is None
                     else [s for s in spans(~ev["present"])])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    style(plt)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.8))
    (a1, a2), (a3, a4) = axes
    t = np.arange(F)

    a1.plot(t, ang, color=SUBJ, lw=1.6)
    a1.set(xlabel="frame t", ylabel="degrees", xlim=(0, F - 1))
    a1.set_title("(a) rotation angle of R_rel vs seed frame", loc="left", pad=14)

    for j, (lbl, ls) in enumerate((("x", "-"), ("y", "--"), ("z", ":"))):
        a2.plot(t, axis[:, j], color=INK2, lw=1.3, ls=ls)
        a2.annotate(lbl, (F - 1, axis[-1, j] if np.isfinite(axis[-1, j]) else 0),
                    xytext=(3, 0), textcoords="offset points", fontsize=7,
                    color=INK2, va="center")
    a2.set(xlabel="frame t", ylabel="unit axis component", xlim=(0, F - 1))
    a2.set_title("(b) rotation axis (camera frame)", loc="left", pad=14)

    for j, (lbl, ls) in enumerate((("x", "-"), ("y", "--"), ("z", ":"))):
        a3.plot(t, tau[:, j], color=INK2, lw=1.3, ls=ls)
        a3.annotate(lbl, (F - 1, tau[-1, j] if np.isfinite(tau[-1, j]) else 0),
                    xytext=(3, 0), textcoords="offset points", fontsize=7,
                    color=INK2, va="center")
    a3.set(xlabel="frame t", ylabel="centroid displacement", xlim=(0, F - 1))
    a3.set_title("(c) tau_rel components (model units)", loc="left", pad=14)

    a4.plot(t, mag, color=DENSE, lw=1.6)
    a4.set(xlabel="frame t", ylabel="||tau_rel||", xlim=(0, F - 1))
    a4.set_title("(d) centroid displacement magnitude", loc="left", pad=14)

    marks = []
    for ax in (a1, a2, a3, a4):
        marks = mark_axis(ax, ev, ref, F)
    for ax, y in ((a1, ang), (a4, mag)):          # dots where the curve is read
        mk = [m for m in marks if np.isfinite(y[m])]
        ax.scatter(mk, y[mk], s=14, color=INK, zorder=4, linewidths=0)

    hand = [Line2D([], [], color=MUTED, lw=1, ls="-.",
                   label=f"seed / reference frame t = {ref}")]
    if ev["t_c"] is not None:
        hand.append(Line2D([], [], color=INK2, lw=1, ls=":",
                           label=f"Stage C canonical frame t_c = {ev['t_c']}"))
    if ev["boundaries"]:
        hand.append(Line2D([], [], color=DENSE, lw=1, ls="--",
                           label=f"encode window boundary {ev['boundaries']}"))
    hand.append(Line2D([], [], color=MUTED, lw=6, alpha=0.18, label="no valid fit"))
    if ev["_absent"]:
        hand.append(Patch(facecolor="none", edgecolor=MASK, hatch="///",
                          label="object absent (Stage C)"))
    fig.legend(handles=hand, loc="upper center", ncol=min(len(hand), 3),
               bbox_to_anchor=(0.5, 1.10))
    fig.suptitle(f"{c}: provisional camera-frame relative rigid motion "
                 f"(blob incl. halo, pre-SAM2)", y=-0.02, fontsize=8, color=INK)
    fig.tight_layout()
    root = Path(str(cfg["paths"]["drive_root"]))
    out = Path(args.out) if args.out else root / "viz" / "thesis" / c
    out.mkdir(parents=True, exist_ok=True)
    save(fig, out, "fig_rigid_motion")
    plt.close(fig)

    nv = int(ok.sum())
    md = [f"# Rigid motion (provisional) - {c}", "",
          f"- source: `cache/stage_b/{c}.npz` fields `fc_R_rel [{F},3,3]`, "
          f"`fc_tau_rel [{F},3]`, `fc_rel_valid [{F}]`; reference (seed) frame "
          f"t = {ref}; channel = {'branch' if fc.get('branch_active') else 'x0'}",
          f"- valid fits: {nv}/{F} frames; rotation angle max "
          f"{np.nanmax(ang):.2f} deg, median {np.nanmedian(ang):.2f} deg; "
          f"||tau_rel|| max {np.nanmax(mag):.4f}, median {np.nanmedian(mag):.4f} "
          "(D4RT model units)",
          f"- critical frames on the axis: seed = {ref}, t_c = {ev['t_c']}, "
          f"encode window boundaries = {ev['boundaries']}; no-valid-fit spans "
          f"{ev['_invalid']}; object-absent spans {ev['_absent']}",
          "- a step AT a window boundary is a re-gauge artefact suspect, not an "
          "object-motion finding (cross-window rigid fits need segment_scales)",
          "- frame: CAMERA frame, per-frame Kabsch of the quiet blob against the "
          "seed frame; centroid-anchored (tau = centroid displacement, rule 9)",
          "- PROVISIONAL: the blob includes the bleed halo; the contract re-fits "
          "this downstream on mask-refined dense tracks",
          "- this is RELATIVE motion, not world motion: on branch clips world "
          "motion is composed (camera trajectory o relative motion) and the "
          "trajectory source is an open decision (retargeting SKILL) - not plotted",
          "- the pivot c that tau is anchored to is NOT persisted by Stage B "
          "(`c_rel` holds the blob compactness centre and is not written out); "
          "`scripts/export_rigid_motion_npz.py` recovers and verifies it"]
    (out / "rigid_motion.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
