#!/usr/bin/env python
"""Stage A.5 bake-off (SKILL_stage_a5 §3-§6) — runs AFTER V-probe sign-off
(human-signed 2026-08-06: cam2world, z-depth, OpenCV axes, first-camera
anchor; margins 3.8x/2.6x pose, 2x corner depth).

Per calibration clip: static-track PROXY (§3.1 — a heuristic, NOT Stage B's
classifier), global Sim3 between the D4RT-derived and ViPE trajectories
(§4.1, with the quasi-static degeneracy fallback), the per-frame scale curve
sigma_t via dense depth ratio (§4.2), and metrics M1-M5 (+M6 cost from the
V-probe). Writes <out_dir>/bakeoff_report.md + plots + calibration record.

READ-ONLY w.r.t. Stage A (§10.1). Recommendation only — the §7 adopt/narrow/
reject decision is applied by the human.

    python scripts/run_bakeoff.py            # all clips in clips.calibration_set

Extension (2026-10-04, Module 3 prerequisite D-M3-07: ViPE trajectory + sigma_t
for clips the bake-off never saw). The A.5 decision is NOT re-opened; the run
only ADDS clips to the archived calibration record and never overwrites an
existing entry unless --force-record is given. Report goes to a separate file.

    python scripts/run_bakeoff.py --clips 000000000404.1.004 000000000265.2.005 000000000370.1.008
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess.bundle import read_bundle          # noqa: E402
from src.preprocess.config import load_stage_a5        # noqa: E402
from src.preprocess.exr import read_exr_z              # noqa: E402


# ---------------------------------------------------------------- primitives
def umeyama_sim3(src: np.ndarray, dst: np.ndarray):
    """Closed-form (s, R, t) minimizing ||dst - (s R src + t)||^2 (§4.1)."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = sc.T @ dc / len(src)
    u, d, vt = np.linalg.svd(cov)
    sgn = np.ones(3)
    if np.linalg.det(vt.T @ u.T) < 0:
        sgn[-1] = -1.0
    R = vt.T @ np.diag(sgn) @ u.T
    var = (sc ** 2).sum() / len(src)
    s = float((d * sgn).sum() / max(var, 1e-12))
    t = mu_d - s * (R @ mu_s)
    return s, R, t


def theil_sen(y: np.ndarray, x: np.ndarray | None = None):
    """Robust slope + bootstrap 95% CI (median of pairwise slopes)."""
    y = np.asarray(y, np.float64)
    x = np.arange(len(y), dtype=np.float64) if x is None else np.asarray(x, np.float64)
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(y) < 3:
        return float("nan"), (float("nan"), float("nan"))
    ii, jj = np.triu_indices(len(y), k=1)
    slopes = (y[jj] - y[ii]) / (x[jj] - x[ii])
    est = float(np.median(slopes))
    rng = np.random.default_rng(0)
    boots = [float(np.median(slopes[rng.integers(0, len(slopes), len(slopes))]))
             for _ in range(200)]
    return est, (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def rot_angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic angle between two rotations, radians."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


# ---------------------------------------------------------------- ViPE side
def load_vipe(vipe_dir: Path, T: int) -> dict:
    """Poses canonicalized to frame-0 camera (P0^-1 P_t), K4, depth + mask
    resampled later per use. Conventions per the signed V-probe."""
    import cv2
    name = next(iter((vipe_dir / "pose").glob("*.npz"))).stem
    pz = np.load(vipe_dir / "pose" / f"{name}.npz")
    poses, inds = pz["data"].astype(np.float64), pz["inds"]
    assert list(inds[:T]) == list(range(T)), "V3 contiguity violated for this clip"
    P0inv = np.linalg.inv(poses[0])
    poses = np.stack([P0inv @ poses[i] for i in range(T)])  # frame-0 camera frame
    K4 = np.load(vipe_dir / "intrinsics" / f"{name}.npz")["data"][0].astype(np.float64)

    depths = {}
    with zipfile.ZipFile(vipe_dir / "depth" / f"{name}.zip") as z:
        for fn in sorted(z.namelist()):
            fi = int(fn.split(".")[0])
            if fi >= T:
                continue
            depths[fi] = read_exr_z(z.read(fn))
    masks = {}
    mz = vipe_dir / "mask" / f"{name}.zip"
    if mz.exists():
        with zipfile.ZipFile(mz) as z:
            for fn in sorted(z.namelist()):
                fi = int(fn.split(".")[0])
                if fi < T:
                    buf = np.frombuffer(z.read(fn), np.uint8)
                    masks[fi] = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    return {"poses": poses, "K4": K4, "depths": depths, "masks": masks}


# ------------------------------------------------------------ §3.1 the proxy
def static_proxy(b: dict, cfg: dict, vipe: dict) -> np.ndarray:
    """Indices of static-PROXY tracks: lowest excursion quantile among tracks
    with >=80% validity, spatially deduplicated, intersected with ViPE's
    static (unmasked) region at the spawn frame. A heuristic ranking — NOT
    Stage B's static/dynamic classifier (§10.4)."""
    X0, Q, anchors = b["X0"], b["Q"], b["anchors"]
    N, T = Q.shape
    t0 = anchors[:, 2].astype(int)
    ref = X0[np.arange(N), t0]
    exc = np.where(Q, np.linalg.norm(X0 - ref[:, None, :], axis=-1), np.nan)
    E = np.nanmax(exc, axis=1)
    eligible = Q.mean(1) >= 0.8
    if eligible.sum() < 20:
        eligible = Q.mean(1) >= 0.5  # degraded-eligibility fallback, recorded
    q = np.nanpercentile(E[eligible], 100 * cfg["align"]["static_proxy_quantile"])
    idx = np.flatnonzero(eligible & (E <= q))
    # ViPE-mask intersection at the spawn frame (native-res mask)
    Hn, Wn = next(iter(vipe["depths"].values())).shape
    Hp = int(cfg["video"]["process_height"])
    keep = []
    for i in idx:
        m = vipe["masks"].get(int(t0[i]))
        if m is None:
            keep.append(i)
            continue
        u = int(anchors[i, 0] * Wn / Hp)   # process->native (anisotropic)
        v = int(anchors[i, 1] * Hn / Hp)
        if m[min(v, Hn - 1), min(u, Wn - 1)] == 0:
            keep.append(i)
    idx = np.asarray(keep, int)
    # spatial dedup on a 16 px grid at the spawn pixel; skip dedup when it
    # would starve the estimators (low-excursion tracks cluster spatially)
    seen, dedup = set(), []
    for i in idx:
        cell = (int(anchors[i, 0] // 16), int(anchors[i, 1] // 16), int(t0[i]) // 16)
        if cell not in seen:
            seen.add(cell)
            dedup.append(i)
    if len(dedup) < 30:
        dedup = list(idx)
    return np.asarray(dedup, int)


# ---------------------------------------------------- §4.2 sigma_t depth ratio
def sigma_curve(b: dict, vipe: dict, cfg: dict) -> dict:
    """Per-frame scale sigma_t = median(D_vipe / D_d4rt) over static pixels,
    dense (D4RT's D is dense). Exclusions: ViPE dynamic mask, far field (top
    5% depth), depth discontinuities (top 10% gradient), non-finite. Deviation
    from spec, recorded: the C >= sigma_conf_min gate is per-track, not
    per-pixel, and C saturates ~0.999 on this checkpoint (contract v1 §5.4),
    so it is replaced by the mask/gradient exclusions above."""
    import cv2
    D = b["D"].astype(np.float32)          # [T, Hp, Wp] z-depth, process res
    T, Hp, Wp = D.shape
    sig, n_valid = np.full(T, np.nan), np.zeros(T, int)
    for t in range(T):
        dv = vipe["depths"].get(t)
        if dv is None:
            continue
        dvp = cv2.resize(dv, (Wp, Hp), interpolation=cv2.INTER_NEAREST)
        m = vipe["masks"].get(t)
        dyn = (cv2.resize(m, (Wp, Hp), interpolation=cv2.INTER_NEAREST) > 0
               if m is not None else np.zeros((Hp, Wp), bool))
        dd = D[t]
        gx = np.abs(np.gradient(dvp, axis=1)) + np.abs(np.gradient(dvp, axis=0))
        ok = (np.isfinite(dvp) & np.isfinite(dd) & (dvp > 1e-6) & (dd > 1e-6)
              & ~dyn
              & (dvp <= np.nanpercentile(dvp, 95))
              & (gx <= np.nanpercentile(gx, 90)))
        n_valid[t] = int(ok.sum())
        if n_valid[t] >= 500:
            sig[t] = float(np.median(dvp[ok] / dd[ok]))
    norm = sig / sig[np.isfinite(sig)][0]
    rate, ci = theil_sen(norm)
    fin = norm[np.isfinite(norm)]
    lin_pred = fin[0] + rate * np.arange(len(fin))
    ss_res = float(((fin - lin_pred) ** 2).sum())
    ss_tot = float(((fin - fin.mean()) ** 2).sum())
    return {"sigma_abs": sig, "sigma_norm": norm, "n_valid": n_valid,
            "low_conf_frames": [int(t) for t in np.flatnonzero(n_valid < 500)],
            "drift_rate_per_frame": rate, "drift_rate_ci95": ci,
            "linear_r2": round(1.0 - ss_res / max(ss_tot, 1e-12), 3)}


# ------------------------------------------------------------------- metrics
def clip_metrics(b: dict, vipe: dict, proxy: np.ndarray, cfg: dict, sig: dict) -> dict:
    import cv2  # noqa: F401
    G, K, X0, Q, P, anchors = b["G"], b["K"], b["X0"], b["Q"], b["P"], b["anchors"]
    T = G.shape[0]
    Hp = int(cfg["video"]["process_height"])
    Hn, Wn = next(iter(vipe["depths"].values())).shape

    oD = G[:, :3, 3].astype(np.float64)
    oV = vipe["poses"][:T, :3, 3]

    # §4.1 global Sim3 (+ degeneracy guard on relative path length)
    scene = float(np.nanmedian([np.nanmedian(v) for v in list(vipe["depths"].values())[:T]]))
    path_rel = float(np.linalg.norm(np.diff(oV, axis=0), axis=1).sum() / max(scene, 1e-9))
    if path_rel >= cfg["align"]["min_baseline"]:
        s_fit, R_fit, t_fit = umeyama_sim3(oD, oV)
        degenerate = False
    else:
        # fallback (§4.1): scale from sigma_0, rotation identity (same axes,
        # both anchored at frame-0 camera), translation from centroid match
        s_fit = float(sig["sigma_abs"][np.isfinite(sig["sigma_abs"])][0])
        R_fit = np.eye(3)
        t_fit = oV.mean(0) - s_fit * oD.mean(0)
        degenerate = True
    rms = float(np.sqrt(np.mean(np.sum(
        (oV - (s_fit * (oD @ R_fit.T) + t_fit)) ** 2, axis=1))))

    sn = np.where(np.isfinite(sig["sigma_norm"]), sig["sigma_norm"], 1.0)
    sa = np.where(np.isfinite(sig["sigma_abs"]), sig["sigma_abs"],
                  np.nanmedian(sig["sigma_abs"]))

    # M2 static-drift flatness under three treatments (relative units: /depth)
    def drift_slope(X):  # X: [n,T,3] proxy tracks in some frame
        t0 = anchors[proxy, 2].astype(int)
        ref = X[np.arange(len(proxy)), t0]
        d = np.linalg.norm(X - ref[:, None, :], axis=-1)
        d = np.where(Q[proxy], d, np.nan)
        z = np.nanmedian(np.abs(X[..., 2]))
        curve = np.nanmedian(d, axis=0) / max(z, 1e-9)
        slope, ci = theil_sen(curve)
        return slope, ci
    Xp = X0[proxy].astype(np.float64)
    m2 = {"raw_X0": drift_slope(Xp),
          "sigma_corrected": drift_slope(Xp * sn[None, :, None]),
          "vipe_anchored": drift_slope(
              (Xp * sa[None, :, None]) @ R_fit.T + t_fit[None, None, :])}

    # M3 static reprojection vs observed track pixels (median px, process res)
    px_err = {"derived_G": [], "vipe_anchor": []}
    K4 = vipe["K4"]
    for t in range(T):
        m = Q[proxy, t]
        if m.sum() < 10:
            continue
        Xw = X0[proxy][m, t].astype(np.float64)
        obs = P[proxy][m, t].astype(np.float64)
        # (i) derived-G: back to camera-t, project with Stage A K
        Xl = (Xw - G[t, :3, 3]) @ np.linalg.inv(G[t, :3, :3]).T
        pr = Xl @ K[t].astype(np.float64).T
        pr = pr[:, :2] / np.clip(pr[:, 2:3], 1e-9, None)
        px_err["derived_G"].append(np.linalg.norm(pr - obs, axis=1))
        # (ii) ViPE anchor: map into ViPE frame (Sim3 + sigma_t), through its
        # canonicalized pose, project with its K, rescale to process px
        Xv = sa[t] * (Xw @ R_fit.T) + t_fit
        Pv = vipe["poses"][t]
        Xc = (Xv - Pv[:3, 3]) @ Pv[:3, :3]      # inv(R) = R.T for c2w pose
        good = Xc[:, 2] > 1e-6
        pu = Xc[good, 0] / Xc[good, 2] * K4[0] + K4[2]
        pv = Xc[good, 1] / Xc[good, 2] * K4[1] + K4[3]
        prv = np.stack([pu * Hp / Wn, pv * Hp / Hn], 1)  # native -> process px
        e = np.full(int(m.sum()), np.nan)
        e[good] = np.linalg.norm(prv - obs[good], axis=1)
        px_err["vipe_anchor"].append(e)
    m3 = {k: (float(np.nanmedian(np.concatenate(v))) if v else float("nan"))
          for k, v in px_err.items()}

    # M4 trajectory smoothness: angular velocity + translational jerk medians
    def smooth(Rs, os):
        ang = [rot_angle(Rs[t], Rs[t + 1]) for t in range(T - 1)]
        jerk = np.linalg.norm(np.diff(os, n=2, axis=0), axis=1)
        return {"ang_vel_med_rad": float(np.median(ang)),
                "jerk_med": float(np.median(jerk) / max(scene, 1e-9)),
                "ang": ang, "jerk": list(jerk)}
    m4 = {"derived_G": smooth(G[:, :3, :3].astype(np.float64), oD * s_fit),
          "vipe": smooth(vipe["poses"][:T, :3, :3], oV)}

    return {"sim3": {"s": s_fit, "rms": rms, "rms_rel": rms / max(scene, 1e-9),
                     "degenerate_path": degenerate, "path_rel": round(path_rel, 4)},
            "m2": m2, "m3": m3, "m4": m4}


# ---------------------------------------------------------------------- main
def merge_record(path: Path, new: dict, force: bool = False) -> dict:
    """Extension rule: ADD clips to an existing calibration record; never drop or
    silently replace an archived entry (Stage B reads this file). Returns the
    merged record; raises on an existing clip unless force."""
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    clash = sorted(set(old) & set(new))
    if clash and not force:
        raise SystemExit(f"calibration record already holds {clash}; pass --force-record to replace")
    merged = dict(old)
    merged.update(new)
    assert set(old) <= set(merged), "merge dropped an archived clip"
    return merged


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", default=None,
                    help="extension run on these clip ids (ViPE cache must exist from probe_vipe.py); "
                         "default = clips.calibration_set (the original bake-off)")
    ap.add_argument("--force-record", action="store_true",
                    help="allow an extension run to replace an existing record entry")
    args = ap.parse_args(argv)
    cfg = load_stage_a5()
    out_root = Path(cfg["paths"]["out_dir"])
    extension = args.clips is not None
    clip_ids = list(args.clips) if extension else list(cfg["clips"]["calibration_set"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    md = ["# Stage A.5 bake-off report (§6)\n" if not extension else
          "# Stage A.5 calibration-record EXTENSION (Module 3 prerequisite, D-M3-07)\n",
          "V-probe signed 2026-08-06: ViPE = cam2world, z-depth, OpenCV axes, "
          "first-camera anchor. Static proxy = excursion heuristic, NOT Stage "
          "B's classifier. Recommendation only — §7 is applied by the human.\n"]
    if extension:
        md += ["A.5 decision (adopt ViPE narrowly, human 2026-08-06) STANDS; this run only adds "
               f"{clip_ids} to the archived record. Metrics are reported for the record, not re-decided.\n"]
    agg = {}
    for clip_id in clip_ids:
        if not (out_root / clip_id / "vipe" / "pose").exists():
            raise SystemExit(f"no ViPE cache for {clip_id}: run scripts/probe_vipe.py --clip <video> first")
        b = read_bundle(cfg, clip_id)
        T = b["G"].shape[0]
        vipe = load_vipe(out_root / clip_id / "vipe", T)
        proxy = static_proxy(b, cfg, vipe)
        sig = sigma_curve(b, vipe, cfg)
        m = clip_metrics(b, vipe, proxy, cfg, sig)
        m1 = b["qc"]["gates"]["A-G2"]["median_rel"]
        agg[clip_id] = (m, sig, m1, len(proxy))

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(sig["sigma_norm"], "-o", ms=2)
        axes[0].axhline(1.0, color="gray", ls=":")
        axes[0].set(title=f"sigma_t (norm) — drift {sig['drift_rate_per_frame']*100:.3f}%/fr "
                          f"CI[{sig['drift_rate_ci95'][0]*100:.3f},{sig['drift_rate_ci95'][1]*100:.3f}] "
                          f"R2={sig['linear_r2']}", xlabel="t")
        for k, (sl, ci) in m["m2"].items():
            axes[1].plot([], [], label=f"{k}: {sl*100:.4f}%/fr")
        axes[1].legend(loc="center")
        axes[1].set_title("M2 drift slopes (rel. units)")
        axes[1].axis("off")
        axes[2].plot(m["m4"]["derived_G"]["ang"], label="derived-G ang vel")
        axes[2].plot(m["m4"]["vipe"]["ang"], label="ViPE ang vel")
        axes[2].set(title="M4 angular velocity [rad/fr]", xlabel="t")
        axes[2].legend()
        fig.tight_layout()
        fig.savefig(out_root / f"bakeoff_{clip_id}.png", dpi=120)
        plt.close(fig)

        md += [f"\n## {clip_id}  (T={T}, proxy n={len(proxy)})\n",
               f"- Sim3: s={m['sim3']['s']:.4f}, center RMS {m['sim3']['rms_rel']:.4f} "
               f"(rel. scene), path_rel={m['sim3']['path_rel']}"
               + (" — **degenerate baseline, fallback path used**" if m['sim3']['degenerate_path'] else ""),
               f"- sigma_t drift: {sig['drift_rate_per_frame']*100:.3f} %/frame, "
               f"CI95 [{sig['drift_rate_ci95'][0]*100:.3f}, {sig['drift_rate_ci95'][1]*100:.3f}], "
               f"R2={sig['linear_r2']}, low-conf frames: {sig['low_conf_frames'] or 'none'}",
               f"- M1 (A-G2 median rel, restated): {m1:.4f}",
               "- M2 drift slope (rel/frame): " + ", ".join(
                   f"{k} {v[0]*100:.4f}%" for k, v in m["m2"].items()),
               f"- M3 static reprojection median px: derived-G {m['m3']['derived_G']:.2f}, "
               f"ViPE-anchored {m['m3']['vipe_anchor']:.2f}",
               f"- M4 ang-vel median rad/fr: derived-G {m['m4']['derived_G']['ang_vel_med_rad']:.4f}, "
               f"ViPE {m['m4']['vipe']['ang_vel_med_rad']:.4f}; jerk (rel): "
               f"{m['m4']['derived_G']['jerk_med']:.5f} vs {m['m4']['vipe']['jerk_med']:.5f}",
               f"- plots: bakeoff_{clip_id}.png"]
        print(f"{clip_id}: done (proxy {len(proxy)}, drift "
              f"{sig['drift_rate_per_frame']*100:.3f}%/fr)")

    # §7 recommendation inputs (stated as recommendation, decided by human)
    m2_gain = [1 - a[0]["m2"]["vipe_anchored"][0] / a[0]["m2"]["raw_X0"][0]
               for a in agg.values() if a[0]["m2"]["raw_X0"][0] > 0]
    m3_gain = [1 - a[0]["m3"]["vipe_anchor"] / a[0]["m3"]["derived_G"]
               for a in agg.values() if a[0]["m3"]["derived_G"] > 0]
    drift_nonzero = [not (a[1]["drift_rate_ci95"][0] <= 0 <= a[1]["drift_rate_ci95"][1])
                     for a in agg.values()]
    md += (["\n## §7 decision inputs\n",
           f"- M2 improvement under ViPE anchor per clip: "
           + ", ".join(f"{g:+.0%}" for g in m2_gain),
           f"- M3 improvement under ViPE anchor per clip: "
           + ", ".join(f"{g:+.0%}" for g in m3_gain),
           f"- sigma_t drift CI excludes zero per clip: {drift_nonzero}",
           "\nRule (§7): adopt-hybrid needs M3 AND M2 >= +30% with nonzero "
           "drift; adopt-narrow if ViPE wins M4 clearly but not M3/M2; else "
           "do-not-adopt. The sigma_t curves + Sim3 records are archived "
           "either way (calibration record).",
           "\n**Decision needed from the human: choose the §7 branch and "
           "record it in the README.**"] if not extension else
           ["\n## No §7 decision in an extension run\n",
            "The §7 branch was chosen 2026-08-06 (README, Stage A.5 outcome). These clips extend "
            "the record for Module 3's camera signal; nothing here re-opens the decision."])
    report_name = "bakeoff_report.md" if not extension else (
        "bakeoff_report_extension_" + "_".join(c[-8:] for c in clip_ids) + ".md")
    (out_root / report_name).write_text("\n".join(md), encoding="utf-8")
    rec = {c: {"sim3_s": a[0]["sim3"]["s"], "sim3_rms_rel": a[0]["sim3"]["rms_rel"],
               "degenerate": a[0]["sim3"]["degenerate_path"],
               "sigma_abs": [None if not np.isfinite(v) else round(float(v), 5)
                             for v in a[1]["sigma_abs"]],
               "drift_rate_per_frame": a[1]["drift_rate_per_frame"],
               "drift_ci95": a[1]["drift_rate_ci95"],
               **({"extension_run": True, "a5_decision": "adopt ViPE narrowly (2026-08-06) unchanged"}
                  if extension else {})}
           for c, a in agg.items()}
    rec_path = out_root / "calibration_record.json"
    if extension:
        rec = merge_record(rec_path, rec, force=args.force_record)
    rec_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\nreport -> {out_root / report_name}")
    print(f"calibration record -> {rec_path}  (clips: {sorted(rec)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
