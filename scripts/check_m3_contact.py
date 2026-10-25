#!/usr/bin/env python
"""Module 3 contact check (reviewer 2026-10-06c). CPU, READ-ONLY: reads cache/m3/<clip>/<category>/lift.npz and
sint_step1.json written by run_m3_sint.py; writes reports/m3_sint/contact.{md,json}. Nothing is re-gauged here.

    python scripts/check_m3_contact.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.anchor.m2c import proxy as m2c_proxy          # noqa: E402  (renderer only)
from src.generate.m3 import contact, handoff as H      # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", default=__import__("os").environ.get("MT_ROOT", "/content/drive/MyDrive/motion_transfer"))
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args(argv)
    lock = H.load_m3_lock()
    for k in ("contact_rows", "contact_band_px", "contact_tol"):
        assert k in lock["int"], f"lock m3.int.{k} missing (reviewer 2026-10-06c proposes 0.03 / 8 / 0.05); John adds it"
    ci = lock["int"]
    m3cfg = __import__("yaml").safe_load(Path("configs/m3.yaml").read_text(encoding="utf-8"))["m3"]
    cc = m3cfg["contact"]; pd_all = m3cfg.get("pair_decisions") or {}
    root = Path(args.drive_root)
    rows = []
    for lp in sorted((root / "cache" / "m3").glob("*/*/lift.npz")):
        pair = f"{lp.parent.parent.name}/{lp.parent.name}"
        s1 = json.loads((lp.parent / "sint_step1.json").read_text(encoding="utf-8"))
        with np.load(lp) as z:
            V, F, Kq = z["V_obj"].astype(np.float64), z["F_obj"].astype(int), z["K_sq"].astype(np.float64)
            Dsrc, mk, am = z["depth_src_sq"].astype(np.float64), z["mask_sq"].astype(bool), z["anchor_mask_sq"].astype(bool)
            Hq, Wq = (int(v) for v in z["grid"])
        zbuf, Mg = m2c_proxy.render_perspective(V, F, Kq, Hq, Wq)
        rc = contact.contact_ratio(zbuf, Mg, Dsrc, mk, am, float(ci["contact_rows"]), int(ci["contact_band_px"]),
                                   int(ci["kbg_block_px"]), args.n_boot, 0)
        od = s1["gates"]["g_obj_depth"]
        plane = contact.fit_ground_plane(Dsrc, mk, am | Mg, Kq, int(cc["plane_nbhd_px"]), int(cc["ransac_iters"]), float(cc["ransac_rel_tol"]),
                                         int(cc["min_inliers"]), int(cc["seed"]))
        rc.update(contact.contact_ratio_plane(zbuf, Mg, plane, Kq, float(ci["contact_rows"]), int(ci["kbg_block_px"]), args.n_boot, 0, anchor_mask=am))
        rc["object_reaches_frame_bottom"] = bool(am[-2:].any())      # the PAINTED object reaches the frame bottom: its lowest visible row may be the frame edge, not a contact line
        rc["mesh_reaches_frame_bottom"] = bool(Mg[-2:].any())               # the rebuilt mesh alone reaching it is a shape matter (edge offset), not truncation
        rc["reading_raw_column"] = contact.reading(rc, od["outcome"] == "fail", float(ci["contact_tol"]))
        rc.update({"pair": pair, "disposition": s1["disposition"], "median_ratio_gate": od["signed_gtar_over_image"], "depth_gate": od["outcome"],
                   "reading": contact.reading_plane(rc, od["outcome"] == "fail", float(ci["contact_tol"])), "instrument": dict(cc),
                   "params": {k: ci[k] for k in ("contact_rows", "contact_band_px", "contact_tol")}})
        dec = contact.decision(rc, float(ci["contact_tol"]), int(lock["lattice"]["px_per_patch"]))
        own = (pd_all.get(pair) or {})
        if own.get("contact") == "regauge_to_ground" and dec["outcome"] == "no_contact_line_in_view" and rc.get("r_c_ground_at_painted_edge"):
            B = rc["r_c_ground_at_painted_edge"]
            dec = {**dec, "s_c": float(1.0 / B), "outcome": "regauge_by_owner_decision", "rule_outcome": "no_contact_line_in_view",
                   "reading": f"painted object reaches the frame bottom, re-gauged to the ground by the owner's decision (configs/m3.yaml): s_c = {1.0 / B:.4f}"}
        if own.get("label"):
            dec["pair_label"] = own["label"]
        rc["decision"] = dec; rc["reading"] = dec["reading"]
        (lp.parent / "contact_decision.json").write_text(json.dumps({"pair": pair, **dec, "decided_by": "pre-registered rule (reviewer 2026-10-06c-e); applied on John's decision 2026-10-06", "unregauged_variant": "run as a sub-row (s_c forced to 1)"}, indent=1), encoding="utf-8")
        rows.append(rc)
        print(f"[m3 contact] {pair}: r_c_plane={rc.get('r_c_plane')} (raw {rc.get('r_c')}) -> {rc['reading']}")
    rep = root / "reports" / "m3_sint"; rep.mkdir(parents=True, exist_ok=True)
    (rep / "contact.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    f = lambda v, n=3: "-" if v is None else f"{v:.{n}f}"          # noqa: E731
    md = ["# Module 3 — object-ground contact check (report only; the per-pair decision is John's at the S-int hard stop)", "",
          f"Corrected column (reviewer 2026-10-06d): a plane is fitted by RANSAC to k_bg x MoGe ground within {cc['plane_nbhd_px']} px of the object "
          f"(inlier tol {cc['ransac_rel_tol']}, at least {cc['min_inliers']} inliers); r_c = G_tar depth at the base's bottom edge / depth where the same ray meets "
          f"that plane. Base = lowest {ci['contact_rows']} of the object's height. Tolerance {ci['contact_tol']} (unchanged).", "",
          "Instrument validation (tests/test_m3_contact.py, standing box, no noise): corrected 1.003 / 1.003 / 1.005 / 0.996 at depth 5 and 9, pitch 0 and 0.25; "
          "raw band column 1.044 / 1.045 / 1.083 / 1.071 on the same scenes.", "",
          "| pair | disp. | median ratio (old gate) | **r_c corrected** | CI95 | base cols | plane inliers (frac) | plane resid | raw band r_c (diagnostic) | upper-body ratio | bottom-edge offset px (G_tar − painted) | ground depth change per row | r_c with ground at the PAINTED edge | reading (corrected column) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ci95 = "-" if not r.get("ci95_plane") else f"[{r['ci95_plane'][0]:.3f}, {r['ci95_plane'][1]:.3f}]"
        inl = "-" if r.get("plane_inliers") is None else f"{r['plane_inliers']} ({r['plane_inlier_frac']:.2f})"
        md.append(f"| {r['pair'][-18:]} | {r['disposition']} | {f(r['median_ratio_gate'])} ({r['depth_gate']}) | **{f(r.get('r_c_plane'))}** | {ci95} | {r.get('n_cols', '-')} | {inl} | "
                  f"{f(r.get('plane_resid_rel_median'), 4)} | {f(r.get('r_c'))} ({r['n_ground_px']} px) | {f(r.get('upper_body_ratio'))} | {f(r.get('bottom_row_offset_px_gtar_minus_anchor'), 1)} | {f(r.get('ground_rel_depth_change_per_row'), 4)} | {f(r.get('r_c_ground_at_painted_edge'))} | {r['reading'] if not r.get('object_reaches_frame_bottom') else 'object cut by the frame bottom: no contact line in view (ratios above are NOT contact readings)'} |")
    md += ["", "A low inlier fraction or a large plane residual means the ground near the object is not one plane in the depth model's output; read that row with care.",
           "", "Edge-offset diagnostic: (offset px) x (depth change per row) is the part of r_c - 1 explained by G_tar's silhouette ending on a different row "
           "from the painted object, with no depth error. The last ratio column removes it. It is a diagnostic; the pre-registered reading is unchanged."]
    (rep / "contact.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
