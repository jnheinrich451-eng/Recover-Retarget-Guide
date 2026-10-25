#!/usr/bin/env python
"""Module 3 — which camera is in the gauge of the lifted scene? (reviewer 2026-10-06f, checks 1-2 and the heading control).
CPU, READ-ONLY. Writes reports/m3_sint/camera_check.{md,json}.

G-int-3 with both cameras: static Stage-A tracks are lifted at t_a (their own camera-t_a coordinates, source gauge) and
reprojected into every later frame of the pair's window through
    D4RT   E^d_t  = G_{0<-t}^-1 G_{0<-t_a}
    ViPE-s E'_t   = bridged with sigma_hat(t_a)     (the rule of record, depth-ratio scale)
    ViPE-u E'_t   = bridged with sim3_s             (the trajectory scale from Umeyama on camera centres)
and compared with where the tracker actually saw them (P, 256^2 pixels). The source video decides: the camera that
reproduces the observed motion of static points is the one in the gauge of the lifted scene.
Also: per-frame rotation disagreement angle(R^d_t, R^ViPE_t); heading of the PURE-D4RT object path vs d_src (control for the
t_a = 47 pairs); sim3_s / sigma_ta. Nothing is decided or changed here.

    python scripts/check_m3_camera.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.generate.m3 import handoff as H                                    # noqa: E402
from src.generate.m3.geometry import camera, vipe_camera                    # noqa: E402
from src.preprocess.stage_c import finalize                                 # noqa: E402

ORTHO_TOL = 1e-4


def in_mask(P_t, M_t):
    Hh, W = M_t.shape
    u = np.rint(P_t[:, 0]).astype(int); v = np.rint(P_t[:, 1]).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < Hh)
    out = np.zeros(len(P_t), bool); out[ok] = M_t[v[ok], u[ok]]
    return out


def reproj_px(X_ta, E_t, K_t):
    Xc = X_ta @ E_t[:3, :3].T + E_t[:3, 3]
    uv = Xc @ K_t.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(Xc[:, 2:3] > 0, uv[:, :2] / uv[:, 2:3], np.nan)


def rot_angle_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1.0) / 2.0, -1.0, 1.0))))


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))))


def run_clip(root: Path, clip: str, cat: str, lock: dict, boundaries_excl: int = 1) -> dict:
    h = json.loads((root / "cache" / "m2_e" / clip / cat / "handoff.json").read_text(encoding="utf-8"))
    t_a = int(h["t_a"]); F_pair, T_l = camera.frames_for_pair(h["n_frames_after_t_a"])
    with np.load(root / "cache" / "stage_a" / clip / "bundle.npz") as z:
        G, K, Xl, P, Q = (z[k] for k in ("G", "K", "X_local", "P", "Q"))
    G, K, Xl, P, Q = G.astype(np.float64), K.astype(np.float64), Xl.astype(np.float64), P.astype(np.float64), Q.astype(bool)
    qa = json.loads((root / "cache" / "stage_a" / clip / "qc.json").read_text(encoding="utf-8"))
    bounds = [b for b in (qa.get("window_boundaries") or []) if t_a < b < t_a + F_pair]
    with np.load(root / "cache" / "stage_b" / f"{clip}.npz") as z:
        y_dyn = z["y_dyn"].astype(bool)
    M, _, _ = finalize.read_mask(root / "cache" / "stage_c" / clip)
    with np.load(sorted((root / "bakeoff" / "stage_a5" / clip / "vipe" / "pose").glob("*.npz"))[0]) as z:
        P_c2w = z["data"].astype(np.float64)
    crec = json.loads((root / lock["trajectory"]["sigma_t_record"]).read_text(encoding="utf-8"))[clip]
    sig_ta, sim3 = float(crec["sigma_abs"][t_a]), float(crec["sim3_s"])
    E_s, _ = vipe_camera.phase2_camera(P_c2w, t_a, sig_ta, F_pair, ORTHO_TOL)
    E_u, _ = vipe_camera.phase2_camera(P_c2w, t_a, sim3, F_pair, ORTHO_TOL)
    Gi = np.linalg.inv(G)
    E_d = np.stack([Gi[t_a + k] @ G[t_a] for k in range(F_pair)])
    static0 = Q[:, t_a] & ~y_dyn & ~in_mask(P[:, t_a], M[t_a])
    X_ta = Xl[:, t_a]
    E_h = E_s.copy(); E_h[:, :3, :3] = E_d[:, :3, :3]                                 # hybrid: D4RT rotation, ViPE (bridged) translation
    E_h2 = E_d.copy(); E_h2[:, :3, 3] = E_s[:, :3, 3]
    per = {"d4rt": [], "vipe_sigma": [], "vipe_sim3": [], "hybrid_rotD4RT_transViPE": []}; rot = []; ns = []; frames = []
    for k in range(1, F_pair):
        t = t_a + k
        ok = static0 & Q[:, t] & ~in_mask(P[:, t], M[t])
        if ok.sum() < 20:
            continue
        frames.append(t); ns.append(int(ok.sum()))
        for name, E in (("d4rt", E_d), ("vipe_sigma", E_s), ("vipe_sim3", E_u), ("hybrid_rotD4RT_transViPE", E_h)):
            e = np.linalg.norm(reproj_px(X_ta[ok], E[k], K[t]) - P[ok, t], axis=1)
            per[name].append(float(np.nanmedian(e)))
        rot.append(rot_angle_deg(E_d[k, :3, :3], E_s[k, :3, :3]))
    # best-fit bridge scale (2026-10-06, after the 035 reading): the scale s that minimises the static points' reprojection error under the
    # ViPE camera, searched on a grid around sigma_ta. A measurement of what the video says the camera's scale is, in source units.
    cands = sig_ta * np.exp(np.linspace(np.log(0.3), np.log(3.0), 61))
    errs = []
    for sc in cands:
        E_c, _ = vipe_camera.phase2_camera(P_c2w, t_a, float(sc), F_pair, ORTHO_TOL)
        e_all = []
        for k in range(1, F_pair, 2):
            t = t_a + k
            ok = static0 & Q[:, t] & ~in_mask(P[:, t], M[t])
            if ok.sum() >= 20:
                e_all.append(np.nanmedian(np.linalg.norm(reproj_px(X_ta[ok], E_c[k], K[t]) - P[ok, t], axis=1)))
        errs.append(float(np.median(e_all)) if e_all else np.nan)
    errs = np.array(errs); ib = int(np.nanargmin(errs))
    best_scale = {"scale": float(cands[ib]), "over_sigma_ta": float(cands[ib] / sig_ta), "over_sim3_s": float(cands[ib] / sim3), "median_px": float(errs[ib]),
                  "px_at_sigma_ta": float(errs[30]), "flat_minimum": bool((errs < errs[ib] * 1.1).sum() > 12)}
    frames = np.array(frames)
    inb = np.zeros(len(frames), bool)
    for b in (qa.get("window_boundaries") or []):
        inb |= np.abs(frames - b) <= boundaries_excl
    seg_before = frames < (min(bounds) if bounds else 10 ** 9)                 # frames in the SAME encoder window as t_a

    def summ(a):
        a = np.array(a)
        return {"median_all": float(np.median(a)), "median_same_window": float(np.median(a[seg_before & ~inb])) if (seg_before & ~inb).any() else None,
                "median_other_windows": float(np.median(a[~seg_before & ~inb])) if (~seg_before & ~inb).any() else None, "last": float(a[-1])}

    # control: heading of the PURE-D4RT object path against d_src (no ViPE anywhere)
    with np.load(root / "cache" / "stage_b5" / f"{clip}.npz") as z:
        Xbar = z["X_bar_track"].astype(np.float64)
    d_src = np.asarray(h["d_src"], np.float64)
    heads = {}
    for w in (4, 8, 12, 16):
        if t_a + w < len(Xbar) and np.isfinite(Xbar[[t_a, t_a + w]]).all():
            heads[str(w)] = angle_deg((Xbar[t_a + w] - Xbar[t_a]) @ G[t_a, :3, :3], d_src)       # R_ta^T (c_{ta+w} - c_ta), camera-t_a frame
    step = np.linalg.inv(G[t_a - 1]) @ G[t_a] if t_a >= 1 else None
    return {"pair": f"{clip}/{cat}", "t_a": t_a, "F_pair": F_pair, "boundaries_in_window": bounds, "n_static_median": int(np.median(ns)) if ns else 0,
            "reproj_px_256": {k: summ(v) for k, v in per.items()} if ns else None,
            "rot_d4rt_vs_vipe_deg": {"median": float(np.median(rot)), "max": float(np.max(rot)), "last": float(rot[-1]), "curve_every8": [round(v, 2) for v in rot[::8]]} if rot else None,
            "best_fit_bridge_scale": best_scale, "sim3_s": sim3, "sigma_ta": sig_ta, "sim3_over_sigma_minus_1": sim3 / sig_ta - 1.0, "sim3_rms_rel": crec["sim3_rms_rel"],
            "d4rt_path_heading_vs_d_src_deg_by_window": heads,
            "d4rt_pose_step_at_t_a": None if step is None else {"rot_deg": rot_angle_deg(np.eye(3), step[:3, :3]), "trans": float(np.linalg.norm(step[:3, 3]))},
            "t_a_on_window_boundary": bool(t_a in (qa.get("window_boundaries") or []))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", default=__import__("os").environ.get("MT_ROOT", "/content/drive/MyDrive/motion_transfer"))
    args = ap.parse_args(argv)
    lock = H.load_m3_lock(); root = Path(args.drive_root)
    seen, rows = set(), []
    for hp in sorted((root / "cache" / "m2_e").glob("*/*/handoff.json")):
        clip, cat = hp.parent.parent.name, hp.parent.name
        if clip in seen:                                   # two targets on one source clip share camera and tracks: one observation
            continue
        seen.add(clip)
        try:
            rows.append(run_clip(root, clip, cat, lock))
        except Exception as e:
            rows.append({"pair": f"{clip}/{cat}", "error": f"{type(e).__name__}: {e}"})
        print(f"[m3 camera] {clip}: {rows[-1].get('error', 'done')}")
    rep = root / "reports" / "m3_sint"; rep.mkdir(parents=True, exist_ok=True)
    (rep / "camera_check.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    f = lambda v, n=2: "-" if v is None else f"{v:.{n}f}"          # noqa: E731
    md = ["# Module 3 — camera check: static tracks lifted at t_a, reprojected through each camera (px at 256², median)", "",
          "One row per source clip. `same window` = frames in t_a's encoder window; `other` = after a window boundary (boundary ±1 frame excluded).", "",
          "| clip | t_a (on boundary?) | static pts | D4RT: all / same / other | ViPE÷σ̂(t_a): all / same / other | ViPE÷sim3_s: all / same / other | rot D4RT-vs-ViPE °: median / max | sim3_s/σ̂−1 | hybrid (D4RT rot + ViPE trans): all | best-fit scale ÷ σ̂ (px there; flat?) | pure-D4RT heading vs d_src ° at 4/8/12/16 frames |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            md.append(f"| {r['pair'][-18:]} | ERROR {r['error'][:160]} |"); continue
        e = r["reproj_px_256"]
        cell = lambda k: "-" if e is None else f"{f(e[k]['median_all'])} / {f(e[k]['median_same_window'])} / {f(e[k]['median_other_windows'])}"   # noqa: E731
        hd = r["d4rt_path_heading_vs_d_src_deg_by_window"]; ro = r["rot_d4rt_vs_vipe_deg"]
        md.append(f"| {r['pair'][:18]} | {r['t_a']} ({r['t_a_on_window_boundary']}) | {r['n_static_median']} | {cell('d4rt')} | {cell('vipe_sigma')} | {cell('vipe_sim3')} | "
                  f"{'-' if ro is None else f(ro['median']) + ' / ' + f(ro['max'])} | {r['sim3_over_sigma_minus_1']:+.3f} | {'-' if e is None else f(e['hybrid_rotD4RT_transViPE']['median_all'])} | {r['best_fit_bridge_scale']['over_sigma_ta']:.2f} ({f(r['best_fit_bridge_scale']['median_px'])}; {r['best_fit_bridge_scale']['flat_minimum']}) | {' / '.join(f(hd.get(k), 1) for k in ('4', '8', '12', '16'))} |")
    md += ["", "Reading (reviewer, pre-stated): the camera with the smaller error on the observed static points is the one in the gauge of the lifted scene. "
           "If ViPE÷sim3_s beats ViPE÷σ̂ on the x0 clips, the translation bridge should use the trajectory scale. A large rotation column means the two cameras "
           "disagree in rotation, which no scale repairs. The last column needs no ViPE: if it matches d_src while the ViPE-composed heading does not, the camera is the cause; "
           "if it misses too, the cause is upstream of Module 3 (t_a on a window boundary, or d_src itself)."]
    (rep / "camera_check.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
