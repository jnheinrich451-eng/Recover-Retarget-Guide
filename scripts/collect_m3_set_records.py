"""Module 3 on the experiment set: one record per frozen pair, assembled from the caches and reports under the experiment root (CPU,
read-only). Written because the batch driver runs one pair at a time and the step reports were overwritten per call until 2026-10-20.

    MT_ROOT=/content/drive/MyDrive/motion_transfer_exp python scripts/collect_m3_set_records.py
    -> <root>/reports/experiment/m3_set_records.{json,md}   (copy into measurements/experiment/)

Per pair: the Module 2 handoff (R2 IoU, alpha, t_a, frames), the lift gates (footprint IoU, depth agreement, k_bg) from
cache/m3/<pair>/sint_step1.json, the geometry (camera source, path length, rotation, object net displacement, coverage) from geometry.npz,
the videos present per arm with their run record (scheduler, corrector-off steps, latent-mask overlap) from reports/m3_s4/guided_runs.json,
and the grader's rows (a0_motion.json; mask-centroid files) when present. Values are copied, never computed here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def jload(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", default=os.environ.get("MT_ROOT", "/content/drive/MyDrive/motion_transfer_exp"))
    ap.add_argument("--handoffs", default="configs/experiment_handoffs.yaml")
    ap.add_argument("--batch", default="configs/m3_batch_exp.yaml")
    args = ap.parse_args(argv)
    root = Path(args.drive_root)
    from scripts.run_m3_batch import video_name
    fz = yaml.safe_load(Path(args.handoffs).read_text(encoding="utf-8"))["experiment_handoffs"]
    cfg = yaml.safe_load(Path(args.batch).read_text(encoding="utf-8")); frames = int(cfg["frames"]); seeds = [int(s) for s in cfg["seeds"]]
    guided = jload(root / "reports" / "m3_s4" / "guided_runs.json") or []
    a0 = jload(root / "reports" / "m3_s4" / "a0_motion.json") or []
    sint2_rep = {r.get("pair"): r for r in (jload(root / "reports" / "m3_sint" / "sint_step2.json") or [])}
    census = {r["clip"]: r for r in (jload(ROOT / "measurements" / "census" / "census.json") or [])}
    import csv as _csv
    contact_labels = {r["clip"]: (r.get("label") or "").strip() or None for r in _csv.DictReader(open(ROOT / "measurements" / "census" / "contact_labels.csv", encoding="utf-8-sig"))} if (ROOT / "measurements" / "census" / "contact_labels.csv").exists() else {}
    bridge = {r.get("pair"): r for r in (jload(root / "reports" / "m3_sint" / "bridge_interval.json") or []) if r.get("camera") == "ba"}
    rows = []
    for pair, role in [(p, "main") for p in fz["main"]] + [(p, "follow_up") for p in (fz.get("follow_up") or {})]:
        clip, cat = pair.split("/"); d = root / "cache" / "m3" / clip / cat
        h = jload(root / "cache" / "m2_e" / clip / cat / "handoff.json") or {}
        s1 = jload(d / "sint_step1.json") or {}
        g = s1.get("gates") or {}
        row = {"pair": pair, "role": role, "t_a": h.get("t_a"), "n_frames_after_t_a": h.get("n_frames_after_t_a"), "handoff_R2_iou": h.get("R2_iou"), "alpha": h.get("alpha"),
               "lift": {"present": (d / "lift.npz").exists(), "footprint_iou": (g.get("g_obj_footprint") or {}).get("iou"), "footprint_outcome": (g.get("g_obj_footprint") or {}).get("outcome"),
                        "depth_rel_diff": (g.get("g_obj_depth") or {}).get("rel_diff"), "depth_outcome": (g.get("g_obj_depth") or {}).get("outcome"),
                        "k_bg": (g.get("g_gauge_scalar") or {}).get("k_bg"), "refused": s1.get("refused") or s1.get("error")}}
        gp = d / "geometry.npz"
        if gp.exists():
            with np.load(gp) as z:
                E = z["E_full"].astype(np.float64); c = z["c_tar"].astype(np.float64); cov = z["cov_curve"].astype(np.float64) if "cov_curve" in z.files else None
                row["geometry"] = {"present": True, "F_pair": int(z["F_pair"]) if "F_pair" in z.files else None, "T_l": int(z["T_l"]) if "T_l" in z.files else None,
                                   "bridge_run": str(z["bridge_run"]) if "bridge_run" in z.files else None, "beta": float(z["beta"]) if "beta" in z.files else None,
                                   "s_delta": float(z["s_delta"]) if "s_delta" in z.files else None, "s_c": float(z["s_c"]) if "s_c" in z.files else None,
                                   "cam_path_len": float(np.linalg.norm(np.diff(np.linalg.inv(E)[:, :3, 3], axis=0), axis=1).sum()),
                                   "cam_rot_total_deg": float(np.degrees(np.arccos(np.clip((E[-1, :3, :3] @ np.array([0, 0, 1.0])) @ np.array([0, 0, 1.0]), -1, 1)))),
                                   "obj_net_disp": float(np.linalg.norm(c[-1] - c[0])), "coverage_first": (None if cov is None else float(cov[min(1, len(cov) - 1)])),
                                   "coverage_last": (None if cov is None else float(cov[-1])), "camera_source": (sint2_rep.get(pair) or {}).get("camera", {}).get("source")}
        else:
            row["geometry"] = {"present": False, "error": (sint2_rep.get(pair) or {}).get("error")}
        vids = {}
        for ak in ["A0", f"A3_{cat}"]:
            for s in seeds:
                name = f"{video_name(ak.split('_')[0], s, frames, list(cfg['arms'][ak]))}.mp4"
                run = next((r for r in guided if r.get("pair") == pair and (r.get("video") or "").endswith(name)), None)
                gr = next((r for r in a0 if r.get("pair") == pair and r.get("video") == name), None)
                vids[name] = {"arm": ak, "present": (d / name).exists(),
                              "run": (None if run is None else {k: run.get(k) for k in ("scheduler", "corrector_off_steps", "latent_mask_overlap", "wall_s", "gpu", "time") if k in run}),
                              "grade": (None if gr is None else {k: gr.get(k) for k in ("continuity", "render_continuity", "bg_disp_median_patch_last", "obj_centroid_disp_last", "expectation_note") if k in gr})}
        row["videos"] = vids
        sup = ((census.get(clip) or {}).get("support") or {})
        br = bridge.get(pair) or {}
        row["beta_record"] = {"beta_used": 1.0, "rho_census_band_ratio": sup.get("ratio"), "rho_ci95": sup.get("ci95"),
                              "s_delta_used": 1.0, "s_delta_six_pair_formula": br.get("s_delta"), "s_delta_formula_inputs": {k: br.get(k) for k in ("Z_tri_c", "Z_L_gnd_c", "rho_c") if k in br},
                              "bridge_record_present": bool(br), "note": "beta = 1 and s_delta = 1 on the set (camera C + BA in the source gauge); rho and the six-pair s_delta recorded beside them (reviewer 2026-10-20)"}
        row["latent_mask_overlap_cells"] = next(((x.get("run") or {}).get("latent_mask_overlap", {}).get("cells") for x in vids.values() if (x.get("run") or {}).get("latent_mask_overlap")), 0)
        qe = jload(root / "cache" / "m2_e" / clip / cat / "qc_m2e.json") or {}; gr = ((qe.get("gate_outcomes") or {}).get("g_gauge_regression") or {})
        r2 = jload(root / "cache" / "m2_e" / clip / cat / "r2.json") or {}; ga = r2.get("gauge_alignment") or {}
        qd = jload(root / "cache" / "m2_d" / clip / cat / "qc_m2d.json") or {}
        k_reg = gr.get("k_regression", ga.get("k_regression"))
        k_np = gr.get("k_regression_nonpositive", ga.get("k_regression_nonpositive"))
        if k_np is None and k_reg is not None:                                        # pairs reconstructed before R1 carry k_regression but not the flag: the flag IS its sign
            k_np = bool(float(k_reg) <= 0)
        row["flags"] = {"kreg_nonpositive": k_np, "k_regression": k_reg,
                        "contact_label": contact_labels.get(clip), "heading_unchecked": ((qd.get("selection_v2") or {}).get("heading_flag")),
                        "source": {"kreg_nonpositive": "cache/m2_e/<pair>/qc_m2e.json: gate_outcomes.g_gauge_regression (R1); r2.json gauge_alignment on pairs reconstructed before the qc carried it",
                                   "contact_label": "measurements/census/contact_labels.csv (John, blind, 2026-10-19)", "heading_unchecked": "cache/m2_d/<pair>/qc_m2d.json: selection_v2.heading_flag (H1 ambiguous -> heading unchecked)"}}
        src = next((r for r in a0 if r.get("pair") == pair and r.get("video") == "source"), None)
        row["source_graded"] = src is not None
        row["mask_centroid_files"] = sorted(p.name for p in (root / "reports" / "m3_s4").glob(f"mask_centroid_{clip}_{cat}*.json"))
        both = all(v["present"] for v in vids.values())
        row["outcome"] = ("complete" if both and row["lift"]["present"] and row["geometry"]["present"] else
                          ("refused: lift " + str(row["lift"]["refused"]) if not row["lift"]["present"] else
                           ("refused: geometry " + str(row["geometry"].get("error")) if not row["geometry"]["present"] else "incomplete: " + ", ".join(k for k, v in vids.items() if not v["present"]))))
        if row["outcome"] == "complete" and row["lift"]["footprint_outcome"] == "fail":
            row["outcome"] = "complete, FLAGGED: footprint gate failed"
        rows.append(row)
        print(f"[m3 set] {pair:30s} {row['outcome']}; footprint {row['lift']['footprint_iou']} ({row['lift']['footprint_outcome']}); cam path {row['geometry'].get('cam_path_len')}; graded {row['source_graded']}", flush=True)
    rep = root / "reports" / "experiment"; rep.mkdir(parents=True, exist_ok=True)
    (rep / "m3_set_records.json").write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
    f = lambda v, n=3: "-" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))  # noqa: E731
    md = ["# Module 3 on the experiment set: record per frozen pair (copied from the caches and reports; nothing computed)", "",
          "| pair | role | outcome | t_a / frames | R2 IoU | footprint IoU | depth rel | k_bg | rho (census) | s_delta (six-pair formula) | overlap cells | kreg<=0 | contact label | heading unchecked | cam path | rot deg | obj net | cov first / last | A0 | A3 | graded |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        g_ = r["geometry"]; v_ = list(r["videos"].values())
        md.append(f"| {r['pair']} | {r['role']} | {r['outcome']} | {r['t_a']} / {r['n_frames_after_t_a']} | {f(r['handoff_R2_iou'])} | {f(r['lift']['footprint_iou'])} {r['lift']['footprint_outcome'] or ''} | {f(r['lift']['depth_rel_diff'])} {r['lift']['depth_outcome'] or ''} | {f(r['lift']['k_bg'])} | "
                  f"{f(r['beta_record'].get('rho_census_band_ratio'))} | {f(r['beta_record'].get('s_delta_six_pair_formula'))} | {r.get('latent_mask_overlap_cells', 0)} | {r['flags'].get('kreg_nonpositive')} | {r['flags'].get('contact_label') or '-'} | {r['flags'].get('heading_unchecked')} | {f(g_.get('cam_path_len'))} | {f(g_.get('cam_rot_total_deg'), 1)} | {f(g_.get('obj_net_disp'))} | {f(g_.get('coverage_first'), 2)} / {f(g_.get('coverage_last'), 2)} | "
                  f"{'yes' if v_ and v_[0]['present'] else 'no'} | {'yes' if len(v_) > 1 and v_[1]['present'] else 'no'} | {'yes' if r['source_graded'] else 'no'} |")
    (rep / "m3_set_records.md").write_text("\n".join(md) + "\n", encoding="utf-8"); print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
