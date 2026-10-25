"""m2_e runner — SKILL m2_e v2.2. Order (§1): band alpha -> delta -> object alpha vs the delta-corrected proxy -> fallback -> R2 -> G_tar check.

  python scripts/run_m2e.py --alpha --all                      # base env: segmenter + Depth-Anything; writes alpha.json (band alpha)
  python scripts/run_m2e.py --delta --all                      # vehicles: silhouette scan + delta-corrected object alpha + fallback (CPU);
                                                               #   humans: needs r2 first (keypoints) — skipped with a note until then
  python scripts/run_m2e.py --r2 --all                         # vehicles: sam3d-objects env, alpha-scaled point map; humans: base env (Body) + post-R2 gauge alignment
  python scripts/run_m2e.py --delta --all                      # again, for the humans (keypoints now exist) and the G_tar cross-check on vehicles
  python scripts/run_m2e.py --qc --all                         # qc_m2e.json per pair + accepted-set summary (labelled pairs excluded)

Per pair, per RETAINED candidate (all K' seeds, the selected one first): the anchor's object mask (same instrument as m2_d step 6;
IoU against M_obj must reproduce the recorded obj_fidelity to 1e-3 or the pair is refused), Depth-Anything disparity, the
ground-band alpha fit with the object-mask fit alongside, intervals, gates -> cache/m2_e/<clip>/<cat>/alpha.json (+ masks, zeta_hat).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.anchor import m2a                                                 # noqa: E402
from src.anchor.m2b2 import cache                                          # noqa: E402
from src.anchor.m2c import compose, placement                              # noqa: E402
from src.anchor.m2d import instrument                                      # noqa: E402
from src.anchor.m2c import proxy as _proxy                                # noqa: E402
from src.anchor.m2e import alpha as A, delta as D, qc as Q                 # noqa: E402

ALPHA_SCHEMA = "m2e_alpha_v1"


def load_ctx(args):
    import yaml
    from src.preprocess.config import load_config
    cfg_a = load_config(args.config); drive = str(cfg_a["paths"]["drive_root"])
    sub = lambda s: Path(str(s).replace("${paths.drive_root}", drive))                       # noqa: E731
    m2e = yaml.safe_load(Path("configs/m2e.yaml").read_text(encoding="utf-8"))
    m2d = yaml.safe_load(Path("configs/m2d.yaml").read_text(encoding="utf-8"))
    th = cfg_a.get("thresholds", {})
    for k in ("m2e_alpha_pixels_min", "m2e_delta_width_max", "m2e_alpha_ci_rel_max"):
        assert k in th, f"{k} missing from calibration.lock.yaml (the human adds it; null = record only)"
    return {"cfg_a": cfg_a, "drive": drive, "cfg": m2e["m2e"], "th": th, "targets": list(m2d["m2d"]["targets"]),
            "cache_m2c": sub(yaml.safe_load(Path("configs/m2c.yaml").read_text(encoding="utf-8"))["paths"]["cache_m2c"]),
            "cache_m2d": sub(m2d["paths"]["cache_m2d"]), "cache_m2e": sub(m2e["m2e"]["paths"]["cache_m2e"]),
            "m2a_out": sub(yaml.safe_load(Path("configs/m2a.yaml").read_text(encoding="utf-8"))["paths"]["out_dir"]),
            "config_hash": hashlib.sha256(json.dumps({k: v for k, v in m2e["m2e"].items() if k not in ("paths", "handoff")}, sort_keys=True).encode()).hexdigest()[:16]}   # handoff = the human's verdicts, not a parameter


def pair_inputs(clip: str, category: str, ctx: dict) -> dict:
    """Everything the alpha step needs for one pair; refuses on any missing or stale upstream."""
    from src.preprocess.stage_c import finalize
    d_m2d = ctx["cache_m2d"] / clip / category; d_m2c = ctx["cache_m2c"] / clip / category
    qc = json.loads((d_m2d / "qc_m2d.json").read_text(encoding="utf-8"))
    assert qc.get("selected_seed") is not None, f"{clip}/{category}: m2_d selected no anchor (g_anchor_selected)"
    fid = json.loads((d_m2d / "fidelity.json").read_text(encoding="utf-8")); gen = json.loads((d_m2d / "generation.json").read_text(encoding="utf-8"))
    arrays, qc3 = compose.read_control(d_m2c); assert arrays is not None
    rec = placement.read_placement(d_m2c); st0 = m2a.read_stage0(ctx["m2a_out"] / clip)
    K_a = st0["K_a"].astype(np.float64); H, W = arrays["M_obj"].shape
    K_img = K_a.copy(); K_img[:2] *= W / 256.0                                 # K_1024 = diag(4, 4, 1) K_256: the control's own camera
    M, present, _ = finalize.read_mask(Path(ctx["cfg_a"]["paths"]["stage_c_cache"]) / clip)
    t_a = int(st0["t_a"])
    import cv2
    hole = cv2.resize(M[t_a].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)   # C2's filled hole = the source subject
    key = f"{clip}/{category}"; disp = ctx["cfg"]["dispositions"]
    if ctx.get("auto") and not (key in disp["accepted"] or key in disp["labelled"]):
        disposition = "experiment"; label = None                                   # the experiment set: admitted by the gates alone, no human disposition (plan v1)
    else:
        assert key in disp["accepted"] or key in disp["labelled"], f"{key}: no disposition in configs/m2e.yaml (SKILL §0)"
        disposition = "accepted" if key in disp["accepted"] else "labelled"; label = disp["labelled"].get(key)
    return {"clip": clip, "category": category, "d_m2d": d_m2d, "qc_m2d": qc, "fidelity": fid, "generation": gen, "control": arrays, "qc_c3": qc3,
            "placement": rec, "stage0": st0, "K_img": K_img, "hole": hole, "t_a": t_a, "disposition": disposition, "label": label,
            "upstream_hashes": {**gen["upstream_hashes"], "m2d_qc": cache.sha256_file(d_m2d / "qc_m2d.json"), "m2d_fidelity": cache.sha256_file(d_m2d / "fidelity.json")}}


def run_alpha(pairs: list, ctx: dict) -> int:
    import cv2
    from scripts.probe_depth_estimator import estimate, load_estimator
    from src.anchor.m2d import segment as _seg
    cfg, th = ctx["cfg"], ctx["th"]; ac = cfg["alpha"]
    sj = Path(ctx["drive"]) / "probe" / "m2d" / "estimator_sign" / "estimator_sign.json"
    es = json.loads(sj.read_text(encoding="utf-8")); assert es["sign"] == 1 and es["estimator"] == cfg["estimator"]["hf_repo"], es
    est, _ = load_estimator(es["estimator"]); seg = _seg.load_segmenter()
    for it in pairs:
        out = ctx["cache_m2e"] / it["clip"] / it["category"]; out.mkdir(parents=True, exist_ok=True)
        ctrl = it["control"]; zeta_ctrl = ctrl["zeta_ctrl"].astype(np.float64); M_obj = ctrl["M_obj"].astype(bool); valid = ctrl["valid"].astype(bool)
        H, W = M_obj.shape; n_plane = np.asarray(it["stage0"]["n_plane"], np.float64)
        fid_by_seed = {c["seed"]: c for c in it["fidelity"]["candidates"]}
        order = [it["qc_m2d"]["selected_seed"]] + [c["seed"] for c in it["generation"]["calls"] if c["seed"] != it["qc_m2d"]["selected_seed"]]
        per, masks, zhs = [], {}, {}
        for k in order:
            f = it["d_m2d"] / "candidates" / f"seed_{k}.png"
            img = np.ascontiguousarray(cv2.imread(str(f))[..., ::-1]); assert img.shape[:2] == (H, W)
            mask, sm = _seg.segment(seg, img, it["category"])
            iou = float((mask & M_obj).sum() / max((mask | M_obj).sum(), 1))
            rec_iou = float(fid_by_seed[k]["obj_fidelity"])
            assert abs(iou - rec_iou) <= float(ac["mask_iou_tol"]), \
                f"{it['clip']}/{it['category']} seed {k}: recomputed mask IoU {iou:.4f} != recorded obj_fidelity {rec_iou:.4f} (instrument did not reproduce m2_d's mask)"
            zh = float(es["sign"]) * estimate(est, img, H, W)
            masks[f"mask_seed_{k}"] = mask; zhs[f"zeta_hat_seed_{k}"] = zh.astype(np.float32)
            if int((mask & M_obj).sum()) < 3:                                                 # the candidate's subject is not where the proxy is (or was not found):
                assert k != order[0], f"{it['clip']}/{it['category']}: the SELECTED seed {k} has no object fit region (mask IoU {iou:.4f}); selection admitted an anchor whose subject is not on the proxy"
                per.append({"seed": k, "selected": False, "mask_iou_vs_M_obj": iou, "segmenter": sm, "band": None, "alpha": None, "b": None, "alpha_source": None,
                            "flag": "object_region_empty", "note": "no fit: anchor mask & M_obj has < 3 pixels (recorded; a non-selected candidate)"})
                print(f"[m2e] {it['clip']}/{it['category']} seed {k}: mask IoU {iou:.4f}; no object fit region -> recorded, not fitted", flush=True); continue
            rho_full = instrument.spearman(zh, zeta_ctrl, mask=valid & ~M_obj)              # = m2_d's bg_fidelity domain: the sign gate (review item 2)
            band, bprov = A.ground_band(mask, M_obj, it["K_img"], n_plane, valid, it["hole"], int(ac["band_inner_px"]), float(ac["band_outer_frac"]))
            fit = A.fit_alpha(zh, zeta_ctrl, band, mask & M_obj, th, blocks=tuple(ac["bootstrap_blocks"]), n_boot=int(ac["n_boot"]))
            fit["g_estimator_sign"] = {"spearman_full_frame": rho_full, "outcome": "pass" if rho_full > 0 else "fail"}
            if rho_full <= 0:
                fit["alpha"], fit["b"], fit["alpha_source"], fit["flag"] = None, None, None, "estimator_sign_negative"
            per.append({"seed": k, "selected": k == order[0], "mask_iou_vs_M_obj": iou, "segmenter": sm, "band": bprov, **fit})
            print(f"[m2e] {it['clip']}/{it['category']} seed {k}{' *' if k == order[0] else ''}: mask IoU {iou:.4f} (rec {rec_iou:.4f}); band {bprov['n_pixels']} px; "
                  f"alpha band {fit['band_fit']['alpha']:.4f} [{fit['band_bootstrap']['block_32']['ci'][0]:.4f}, {fit['band_bootstrap']['block_32']['ci'][1]:.4f}] "
                  f"obj(uncorr) {fit['object_fit_uncorrected']['alpha']:.4f}; naive/reversed band {fit['band_fit']['naive_over_reversed']:.3f}; rho_full {rho_full:+.3f}; source {fit['alpha_source']} {fit['flag'] or ''}", flush=True)
        sel = per[0]
        alphas = [p["alpha"] for p in per if p["alpha"] is not None]
        rec = {"schema": ALPHA_SCHEMA, "clip": it["clip"], "category": it["category"], "disposition": it["disposition"], "label": it["label"], "t_a": it["t_a"],
               "K_1024": it["K_img"].tolist(), "fx_over_fy": float(it["K_img"][0, 0] / it["K_img"][1, 1]),
               "fit_region": {"primary": "ground_band", "check": "anchor_mask & M_obj", **sel["band"], "band_inner_px": int(ac["band_inner_px"]), "band_outer_frac": float(ac["band_outer_frac"])},
               "fit_direction": A.FIT_DIRECTION, "estimator": es["estimator"], "estimator_sign": es["sign"],
               "alpha": sel["alpha"], "b": sel["b"], "alpha_source": sel["alpha_source"], "flag": sel["flag"], "selected_seed": sel["seed"],
               "alpha_per_candidate": {str(p["seed"]): p["alpha"] for p in per}, "alpha_spread": float(np.ptp(alphas)) if len(alphas) > 1 else None,
               "alpha_spread_rel": float(np.ptp(alphas) / np.median(alphas)) if len(alphas) > 1 and np.median(alphas) else None,
               "candidates": per, "config_hash": ctx["config_hash"], "upstream_hashes": it["upstream_hashes"]}
        (out / "alpha.json").write_text(json.dumps(rec, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
        np.savez_compressed(out / "anchor_masks.npz", **{k: v for k, v in masks.items()}, **zhs)
        print(f"[m2e] {it['clip']}/{it['category']} ({it['disposition']}): alpha {rec['alpha']} b {rec['b']} source {rec['alpha_source']}; "
              f"per-candidate {rec['alpha_per_candidate']} spread_rel {rec['alpha_spread_rel']} -> {out / 'alpha.json'}")
    viz = Path(ctx["drive"]) / "viz" / "m2e"; viz.mkdir(parents=True, exist_ok=True)
    L = ["# m2_e step 1 — alpha (ground-band fit, object-mask check)", "",
         "| clip | cat | disp. | alpha (band) | CI32 | CI64 | alpha (object, uncorrected) | band/obj (uncorr.) | naive/reversed (band) | rho_full | n_band | source | flag | per-candidate spread rel |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for it in pairs:
        r = json.loads((ctx["cache_m2e"] / it["clip"] / it["category"] / "alpha.json").read_text(encoding="utf-8")); s = r["candidates"][0]
        ci = s["band_bootstrap"]["block_32"]["ci"]; ci6 = s["band_bootstrap"]["block_64"]["ci"]
        L.append(f"| {it['clip'][-8:]} | {it['category']} | {r['disposition']} | {s['band_fit']['alpha']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | [{ci6[0]:.4f}, {ci6[1]:.4f}] | {s['object_fit_uncorrected']['alpha']:.4f} | "
                 f"{s['band_vs_object_uncorrected_ratio']:.3f} | {s['band_fit']['naive_over_reversed']:.3f} | {s['g_estimator_sign']['spearman_full_frame']:+.3f} | {s['band']['n_pixels']} | "
                 f"{r['alpha_source']} | {r['flag'] or ''} | {r['alpha_spread_rel']} |")
    L += ["", "naive/reversed = Depth-Anything's empirical lambda on this set (reported, not predicted; review item 12).",
          "band/obj (uncorrected) is NOT a flattening diagnostic: the uncorrected object fit absorbs the anchor-vs-proxy yaw (test 3c: ratio 1.42 with no flattening).",
          "The delta-corrected object fit, the flattening flag and the fallback decision are made in the delta step (reviewer 2026-09-23).",
          "Gates m2e_alpha_pixels_min / m2e_alpha_ci_rel_max are null (record only)."]
    (viz / "report_m2e_alpha.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print(f"[m2e] -> {viz / 'report_m2e_alpha.md'}")
    return 0


def _load_alpha(ctx, it):
    f = ctx["cache_m2e"] / it["clip"] / it["category"] / "alpha.json"
    assert f.exists(), f"{it['clip']}/{it['category']}: alpha.json missing — run --alpha first (SKILL §1: alpha before R2)"
    rec = json.loads(f.read_text(encoding="utf-8"))
    assert rec.get("schema") == ALPHA_SCHEMA and rec.get("config_hash") == ctx["config_hash"], "alpha.json is from another config/schema — re-run --alpha"
    assert rec["fit_region"]["primary"] == "ground_band" and rec["fit_direction"] == A.FIT_DIRECTION, "alpha.json predates the band definition — re-run --alpha"
    return rec, f


def _proxy_mesh(ctx, it):
    from src.anchor.m2b2.objects import load_m2b2_config
    b2 = load_m2b2_config("configs/m2b2.yaml", drive_root=ctx["drive"]); variant = cache.route(it["category"], b2)
    prox = cache.read_proxy(Path(b2["paths"]["cache_r1"]) / it["category"] / it["placement"]["proxy_cache_key"], variant, b2["m2b2"]["adapter_version"])
    return prox, variant, b2


def run_delta(pairs: list, ctx: dict) -> int:
    """Vehicles: restricted silhouette scan vs the anchor mask, then the delta-corrected object alpha and the fallback decision;
    G_tar cross-check when g_tar_mesh.npz exists. Humans: keypoint facing from the Body R2 record (skipped until it exists)."""
    dc = ctx["cfg"]["delta"]; th = ctx["th"]
    for it in pairs:
        out = ctx["cache_m2e"] / it["clip"] / it["category"]; rec_a, _ = _load_alpha(ctx, it)
        prox, variant, _ = _proxy_mesh(ctx, it); pl = it["placement"]["placement"]
        X, s, R_A = np.asarray(pl["X"], float), float(pl["s"]), np.asarray(pl["R_A"], float)
        n_plane = np.asarray(it["stage0"]["n_plane"], float); K = it["K_img"]; H, W = it["control"]["M_obj"].shape
        with np.load(out / "anchor_masks.npz") as z:
            amask = z[f"mask_seed_{rec_a['selected_seed']}"].astype(bool); zh = z[f"zeta_hat_seed_{rec_a['selected_seed']}"].astype(np.float64)
        M_obj = it["control"]["M_obj"].astype(bool)
        if variant == "objects":
            sc = D.paired_scan(prox["vertices"], prox["faces"], X, s, R_A, n_plane, K, amask, scan_deg=float(dc["scan_deg"]), step_deg=float(dc["scan_step_deg"]),
                               tol=float(dc["curve_width_tol"]), refine=bool(dc["parabolic_refine"]))
            rec_iou = float(next(c["obj_fidelity"] for c in it["fidelity"]["candidates"] if c["seed"] == rec_a["selected_seed"]))
            assert abs(sc["iou_at_zero"] - rec_iou) < 1e-3, f"theta = 0 IoU {sc['iou_at_zero']:.4f} != m2_d's obj_fidelity {rec_iou:.4f} (same render expected)"
            # the delta-corrected object comparison (reviewer 2026-09-23): the proxy rotated by the measured delta, its disparity over M
            zb, mk = compose.render_object(prox["vertices"], prox["faces"], X, s, D.rot_about(n_plane, sc["delta_deg"]) @ R_A, K, H, W)
            zeta_rot = np.where(mk & (zb > 0), 1.0 / np.where(zb > 0, zb, 1.0), np.nan)
            region = amask & mk
            dec = A.decide_alpha(rec_a["candidates"][0], zh, zeta_rot, region, th, tuple(ctx["cfg"]["alpha"]["bootstrap_blocks"]), int(ctx["cfg"]["alpha"]["n_boot"]))
            gt = out / "g_tar_mesh.npz"; vs_gtar = None
            if gt.exists():
                with np.load(gt) as z:
                    Vg, Fg = z["vertices"].astype(np.float64), z["faces"].astype(int)
                _, mg = _proxy.render_perspective(Vg, Fg, K, H, W)
                vs_gtar = D.yaw_scan(prox["vertices"], prox["faces"], X, s, R_A, n_plane, K, mg, float(dc["scan_deg"]), float(dc["scan_step_deg"]), float(dc["curve_width_tol"]), bool(dc["parabolic_refine"]))
            # g_front_back (reviewer 2026-09-23): a 180-deg flip always changes the 4-class heading; m2_d's blind full-resolution label vs the control class
            lab = next((c["heading_class"] for c in it["qc_m2d"]["candidates"] if c["seed"] == rec_a["selected_seed"]), None)
            ctrl_cls = it["qc_m2d"]["heading_anchor_frame"]["class"]
            g_fb = {"human_label": lab, "control_class": ctrl_cls, "outcome": ("unchecked" if lab is None else ("pass" if lab == ctrl_cls else "fail")), "front_back_margin": sc["front_back_margin"],
                    "label_instrument": next((c.get("heading_instrument") for c in it["qc_m2d"]["candidates"] if c["seed"] == rec_a["selected_seed"]), "human label (heading_labels.csv)")}
            shape = None
            if gt.exists():
                V_posed = X + s * _proxy.aligned_vertices(_proxy.ProxyMesh("proxy", prox["vertices"], prox["faces"], prox["vertices"].mean(0), prox["B_tar"]), R_A)
                shape = D.gtar_yaw_vs_proxy(Vg, V_posed, n_plane)
            f_flat = dec.get("band_vs_object_ratio")
            pred = D.flatten_predicted_yaw(float(f_flat), float(it["qc_m2d"]["heading_anchor_frame"]["azimuth_deg"])) if f_flat and np.isfinite(f_flat) else None
            drec = {"schema": "m2e_delta_v1", "variant": "objects", "delta_deg": sc["delta_deg"], "delta_instrument": "silhouette_scan", "delta_definition": D.DELTA_DEFINITION, "scan": sc,
                    "g_front_back": g_fb, "gtar_shape_yaw": shape, "flatten_predicted_yaw": pred,
                    "delta_vs_g_tar_deg": None if vs_gtar is None else vs_gtar["delta_deg"], "scan_vs_g_tar": vs_gtar,
                    "alpha_decision": {k: dec[k] for k in ("alpha", "b", "alpha_source", "flag", "band_vs_object_ratio", "alpha_object_corrected", "object_fit_corrected", "object_bootstrap_corrected", "gates")},
                    "gates": {"g_delta_scan_peaked": {"value": sc["curve_width_deg"], "threshold": th.get("m2e_delta_width_max"),
                                                      "outcome": "ungated" if th.get("m2e_delta_width_max") is None else ("pass" if sc["curve_width_deg"] <= th["m2e_delta_width_max"] else "fail")}}}
            print(f"[m2e] {it['clip']}/{it['category']}: delta {sc['delta_deg']:+.2f} deg (peak IoU {sc['iou_peak']:.3f}, at 0: {sc['iou_at_zero']:.3f}, width {sc['curve_width_deg']:.0f} deg"
                  f"{', at scan edge' if sc['peak_at_scan_edge'] else ''}; flipped peak {sc['flipped_peak_iou']:.3f}, margin {sc['front_back_margin']:+.3f}; g_front_back {g_fb['outcome']} [{lab} vs {ctrl_cls}])"
                  f"; alpha {dec['alpha']} source {dec['alpha_source']} flag {dec['flag']} band/obj(corrected) {dec['band_vs_object_ratio']:.3f}"
                  + (f"; vs G_tar {vs_gtar['delta_deg']:+.2f}" if vs_gtar else ""))
        else:
            r2f = out / "r2_body.json"
            if not r2f.exists():
                print(f"[m2e] {it['clip']}/{it['category']}: human delta needs the Body R2 record (run --r2 first) — skipped"); continue
            r2 = json.loads(r2f.read_text(encoding="utf-8")); kp = np.asarray(r2["kp3d_cam"], float)
            bd = D.body_delta(kp, R_A, prox["B_tar"], n_plane)
            dec = dict(rec_a["candidates"][0]); dec.setdefault("band_vs_object_ratio", None)
            drec = {"schema": "m2e_delta_v1", "variant": "body", "delta_deg": bd["delta_deg"], "delta_instrument": "keypoints", "delta_definition": D.DELTA_DEFINITION, "body": bd,
                    "g_front_back": {"outcome": "n/a (keypoints read front from back directly)"},
                    "alpha_decision": {"alpha": rec_a["alpha"], "b": rec_a["b"], "alpha_source": rec_a["alpha_source"], "flag": rec_a["flag"]},
                    "gates": {"g_delta_scan_peaked": {"outcome": "n/a (keypoints)"}}}
            print(f"[m2e] {it['clip']}/{it['category']}: delta {bd['delta_deg']:+.2f} deg from keypoints (shoulder/hip twist {bd['shoulder_hip_angle_deg']:.1f} deg)")
        if variant == "objects":                                            # SIGN FIXTURE for the human's eye: the proxy at 0 (magenta) and at +15 deg (green) over the anchor
            import cv2
            _, m0 = compose.render_object(prox["vertices"], prox["faces"], X, s, R_A, K, H, W)
            _, m15 = compose.render_object(prox["vertices"], prox["faces"], X, s, D.rot_about(n_plane, 15.0) @ R_A, K, H, W)
            fx_img = cv2.imread(str(it["d_m2d"] / "candidates" / f"seed_{rec_a['selected_seed']}.png"))
            for m_, col in ((m0, (255, 0, 255)), (m15, (0, 255, 0))):
                e = m_ ^ cv2.erode(m_.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool); fx_img[e] = col
            cv2.putText(fx_img, "magenta = proxy at delta 0; green = proxy at delta +15 deg", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            vz = Path(ctx["drive"]) / "viz" / "m2e"; vz.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(vz / f"{it['clip']}_{it['category']}_sign_fixture.png"), fx_img)
        (out / "delta.json").write_text(json.dumps(drec, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    return 0


def run_r2(pairs: list, ctx: dict) -> int:
    """R2 through m2b2's r2_anchor. Objects: the alpha-scaled point map (never zeta_ctrl; API guard + numerical tripwire), N seeds,
    medoid by centroid-aligned IoU vs the anchor mask. Body: its own depth; then the post-R2 gauge alignment (scale + translation,
    no rotation) of the rendered mesh depth to the band-alpha anchor depth over the object mask (SKILL v2.2 §1)."""
    import cv2
    from src.anchor.m2b2 import pipeline as _pipe
    for it in pairs:
        out = ctx["cache_m2e"] / it["clip"] / it["category"]; rec_a, f_a = _load_alpha(ctx, it)
        prox, variant, b2 = _proxy_mesh(ctx, it)
        dj = out / "delta.json"
        if variant == "objects":
            assert dj.exists(), "vehicles: run --delta before --r2 (the fallback decision fixes alpha)"; dec = json.loads(dj.read_text(encoding="utf-8"))["alpha_decision"]
        else:
            dec = {"alpha": rec_a["alpha"], "b": rec_a["b"], "alpha_source": rec_a["alpha_source"]}
        assert dec["alpha"] is not None and dec["alpha"] > 0, f"{it['clip']}/{it['category']}: alpha undetermined — R2 refused (no point map can be built)"
        with np.load(out / "anchor_masks.npz") as z:
            amask = z[f"mask_seed_{rec_a['selected_seed']}"].astype(bool); zh = z[f"zeta_hat_seed_{rec_a['selected_seed']}"].astype(np.float64)
        img = np.ascontiguousarray(cv2.imread(str(it["d_m2d"] / "candidates" / f"seed_{rec_a['selected_seed']}.png"))[..., ::-1])
        K = it["K_img"]; valid = it["control"]["valid"].astype(bool); M_obj = it["control"]["M_obj"].astype(bool)
        up = {**it["upstream_hashes"], "alpha": cache.sha256_file(f_a), "alpha_source": dec["alpha_source"]}
        if variant == "objects":
            from src.anchor.m2b2.objects import ObjectsReconstructor
            A.pointmap_source_guard("alpha_scaled_zeta_hat")
            band, _ = A.ground_band(amask, M_obj, K, np.asarray(it["stage0"]["n_plane"], float), valid, it["hole"], int(ctx["cfg"]["alpha"]["band_inner_px"]), float(ctx["cfg"]["alpha"]["band_outer_frac"]))
            pm, depth, pm_prov = A.anchor_pointmap(zh, float(dec["alpha"]), float(dec["b"]), K, valid, region=amask | M_obj | band)
            print(f"[m2e] {it['clip']}/{it['category']}: point map positive on the object/band region (non-positive {100 * pm_prov['region_nonpositive_frac']:.1f} %); "
                  f"far field non-positive {100 * pm_prov['far_field_nonpositive_frac']:.1f} % of valid pixels (recorded: the affine bridge's reach)")
            trip = A.assert_not_proxy(depth, it["control"]["zeta_obj"].astype(np.float64), M_obj)
            # what the vendor will see over the object: depth statistics of the alpha-scaled point map inside the anchor mask
            dm = depth[amask]; dm_ok = dm[np.isfinite(dm)]
            obj_depth = {"n_mask": int(amask.sum()), "n_valid": int(len(dm_ok)), "p02": float(np.percentile(dm_ok, 2)), "p50": float(np.percentile(dm_ok, 50)),
                         "p98": float(np.percentile(dm_ok, 98)), "p98_over_p02": float(np.percentile(dm_ok, 98) / max(np.percentile(dm_ok, 2), 1e-12)),
                         "frac_invalid_in_mask": float(1 - len(dm_ok) / max(amask.sum(), 1)), "z_bg": pm_prov["z_bg"],
                         "disparity_p02_in_mask": float(np.percentile((float(dec["alpha"]) * zh + float(dec["b"]))[amask], 2))}
            print(f"[m2e] {it['clip']}/{it['category']}: object depth in the point map: p02 {obj_depth['p02']:.3f} p50 {obj_depth['p50']:.3f} p98 {obj_depth['p98']:.3f} "
                  f"(p98/p02 {obj_depth['p98_over_p02']:.2f}), invalid in mask {100 * obj_depth['frac_invalid_in_mask']:.1f} %, z_bg {obj_depth['z_bg']:.3f}, disparity p02 in mask {obj_depth['disparity_p02_in_mask']:.4f}")
            recon = ObjectsReconstructor(b2)
            # Reviewer 2026-09-27, pre-registered: the sparse-structure stage is a SAMPLER; catch the failure PER SEED, take the medoid
            # over the successful seeds (centroid-aligned IoU vs the anchor mask, ties -> lowest seed), record n_success, refuse only
            # if all N fail. The vendor's occupancy threshold is not touched (that would be loosening, one layer down).
            from src.anchor.m2b2 import silhouette as _sil
            from src.anchor.m2b2.pipeline import _gaussian_arrays
            per_seed, cands = [], []
            for sd in range(int(b2["m2b2"]["seeds"]["r2"])):
                try:
                    r_sd = recon.reconstruct(img, amask, pm.astype(np.float32), "stage_a", sd, K=K)
                    iou_al, shift = _sil.centroid_aligned_iou(r_sd.X_cv, r_sd.faces, K, *amask.shape, amask)
                    per_seed.append({"seed": sd, "ok": True, "iou": r_sd.selfcheck_iou, "iou_aligned": float(iou_al), "mask_centroid_offset_px": list(shift),
                                     "watertight": r_sd.watertight, "n_vertices": int(len(r_sd.X_cv)), "seconds": r_sd.seconds, "t_cv": r_sd.X_cv.mean(0).tolist()})
                    cands.append(r_sd)
                except Exception as e:                                        # noqa: BLE001 — one seed's empty sample is the vendor's variance
                    per_seed.append({"seed": sd, "ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}); cands.append(None)
                print(f"[m2e] {it['clip']}/{it['category']} R2 seed {sd}: {'ok, aligned IoU %.3f' % per_seed[-1]['iou_aligned'] if per_seed[-1]['ok'] else 'FAILED ' + per_seed[-1]['error']}", flush=True)
            n_success = sum(p_["ok"] for p_ in per_seed); fallback = None
            if n_success == 0:
                fail = {"schema": "m2e_r2_failed_v1", "variant": "objects", "error": "all seeds failed: " + per_seed[0]["error"], "per_seed": per_seed, "pointmap": pm_prov,
                        "pointmap_tripwire": trip, "object_depth": obj_depth, "alpha_used": dec["alpha"], "b_used": dec["b"], "alpha_source": dec["alpha_source"]}
                (out / "r2_failed.json").write_text(json.dumps(fail, indent=1), encoding="utf-8")
                print(f"[m2e] {it['clip']}/{it['category']}: R2 FAILED on all {len(per_seed)} seeds with the alpha-scaled point map -> {out / 'r2_failed.json'}")
                # FALLBACK (reviewer 2026-09-27, pre-registered): the vendor's own point map, then the Body-style scale-and-translation
                # alignment of the rendered mesh depth to the band-alpha anchor depth over the mask. G_tar reaches the source gauge without
                # the proxy. Recorded as pointmap_source = "vendor_internal + gauge_align"; never silently the same as the primary path.
                per_seed_v, cands_v = [], []
                for sd in range(int(b2["m2b2"]["seeds"]["r2"])):
                    try:
                        r_sd = recon.reconstruct(img, amask, None, "vendor_internal", sd, K=K)
                        iou_al, shift = _sil.centroid_aligned_iou(r_sd.X_cv, r_sd.faces, K, *amask.shape, amask)
                        per_seed_v.append({"seed": sd, "ok": True, "iou": r_sd.selfcheck_iou, "iou_aligned": float(iou_al), "mask_centroid_offset_px": list(shift), "n_vertices": int(len(r_sd.X_cv)), "seconds": r_sd.seconds})
                        cands_v.append(r_sd)
                    except Exception as e:                                    # noqa: BLE001
                        per_seed_v.append({"seed": sd, "ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}); cands_v.append(None)
                    print(f"[m2e] {it['clip']}/{it['category']} R2 FALLBACK (vendor map) seed {sd}: {'ok, aligned IoU %.3f' % per_seed_v[-1]['iou_aligned'] if per_seed_v[-1]['ok'] else 'FAILED'}", flush=True)
                if not any(p_["ok"] for p_ in per_seed_v):
                    print(f"[m2e] {it['clip']}/{it['category']}: fallback failed too (no r2.json; qc skips this pair)"); continue
                per_seed, cands = per_seed_v, cands_v; n_success = sum(p_["ok"] for p_ in per_seed); fallback = "vendor_internal + gauge_align"
            ious = np.array([p_["iou_aligned"] if p_["ok"] else -1.0 for p_ in per_seed]); idx = int(np.flatnonzero(ious >= ious.max() - 1e-12)[0])
            best = cands[idx]; V, F = np.asarray(best.X_cv, np.float64), np.asarray(best.faces, int); gs = _gaussian_arrays(best.gaussians) or {}
            gauge = None
            if fallback:
                zb_v, mv = _proxy.render_perspective(V, F, K, *M_obj.shape)
                gauge = D.gauge_align(zb_v, depth, mv & amask, mode="median_ratio"); iou_pre = D.iou(mv, amask); V = D.apply_gauge(V, gauge["scale_k"], gauge["shift_t"])
                _, mv2 = _proxy.render_perspective(V, F, K, *M_obj.shape); iou_post = D.iou(mv2, amask)
                assert abs(iou_post - iou_pre) < 1e-6, f"gauge changed the projection: IoU {iou_pre:.6f} -> {iou_post:.6f} (t must be 0 and k about the camera centre)"
                gauge["projection_unchanged"] = {"iou_pre": iou_pre, "iou_post": iou_post}
                for k_ in list(gs):
                    if k_.endswith("xyz"):
                        gs[k_] = D.apply_gauge(np.asarray(gs[k_], np.float64), gauge["scale_k"], gauge["shift_t"]).astype(np.float32)
                pm_prov = {**pm_prov, "pointmap_source": fallback, "primary_failed": fail["error"], "primary_per_seed": fail["per_seed"]}
                print(f"[m2e] {it['clip']}/{it['category']}: fallback gauge k(median ratio) {gauge['scale_k']:.4f} [regression would give k {gauge['k_regression']:.4f} t {gauge['t_regression']:+.4f}] over {gauge['n_pixels']} px")
            gs = {f"gs_{k}": np.asarray(v, np.float32) for k, v in gs.items()}
            cache.write_anchor(out / "r2" / it["clip"], V, F, K, it["t_a"], {"variant": "objects", "per_seed": per_seed, "medoid_seed": idx, "n_success": n_success, "upstream_hashes": up},
                               {k[3:]: v for k, v in gs.items()} or None)
            r = {"medoid_seed": per_seed[idx]["seed"], "qc": {"seeds": [p_["seed"] for p_ in per_seed], "per_seed": per_seed, "mask_centroid_offset_px": per_seed[idx]["mask_centroid_offset_px"],
                                                             "model_version": recon.model_version, "n_success": n_success}}
            np.savez_compressed(out / "g_tar_mesh.npz", vertices=V.astype(np.float32), faces=F.astype(np.int32), K_1024=K.astype(np.float32), t_a=np.int64(it["t_a"]),
                                frame="camera_t_a", gauge="source (alpha-scaled anchor depth)")
            if gs:
                np.savez_compressed(out / "g_tar_gauss.npz", **gs)
            _, mg = _proxy.render_perspective(V, F, K, *M_obj.shape)
            r2rec = {"schema": "m2e_r2_v1", "variant": "objects", "pointmap": pm_prov, "pointmap_tripwire": trip, "r2_seeds": r["qc"]["seeds"], "r2_medoid_seed": r["medoid_seed"],
                     "r2_n_success": r["qc"]["n_success"], "object_depth": obj_depth, "fallback": fallback, "gauge_alignment": gauge,
                     "r2_iou_vs_anchor_mask": float(D.iou(mg, amask)), "r2_mask_centroid_offset_px": r["qc"]["mask_centroid_offset_px"], "gaussians_available": bool(gs),
                     "alpha_used": dec["alpha"], "b_used": dec["b"], "alpha_source": dec["alpha_source"], "per_seed": r["qc"]["per_seed"], "model_version": r["qc"]["model_version"]}
            print(f"[m2e] {it['clip']}/{it['category']}: R2 objects medoid seed {r['medoid_seed']} of {r['qc']['n_success']} successful, IoU vs anchor mask {r2rec['r2_iou_vs_anchor_mask']:.3f}, gaussians {bool(gs)}, tripwire rel resid {trip.get('relative_residual')}")
        else:
            from src.anchor.m2b2.body import BodyReconstructor
            recon = BodyReconstructor(b2)
            try:
                r = recon.reconstruct(img, amask, K)                         # Body is deterministic: one call, its own depth (no point map)
            except AssertionError as e:                                      # the adapter's contract asserts (SKILL §4: principal point within the tolerance after the
                if not ctx.get("auto"):                                      # square-pixel rescale): on the experiment path a RECORDED refusal of the pair, never a loosening
                    raise
                (out / "r2_refused.json").write_text(json.dumps({"schema": "m2e_r2_refused_v1", "variant": "body", "stage": "body_reconstruct",
                                                                   "reason": str(e), "tolerance_px": b2["m2b2"]["body"].get("principal_point_max_px") if isinstance(b2["m2b2"].get("body"), dict) else None,
                                                                   "K_1024": K.tolist()}, indent=1), encoding="utf-8")
                print(f"[m2e] {it['clip']}/{it['category']}: Body R2 refused by the adapter's contract: {e} -> r2_refused.json", flush=True); continue
            V, F = np.asarray(r.X_cv, np.float64), np.asarray(r.faces, int); kp = np.asarray(r.extra["kp3d_cam"], np.float64)
            cache.write_anchor(out / "r2" / it["clip"], V, F, K, it["t_a"], {"variant": "body", "selfcheck_iou": r.selfcheck_iou, "kp_in_mask_frac": r.kp_in_mask_frac,
                                                                            "n_people": r.n_people, "focal_returned": r.focal_returned, "upstream_hashes": up})
            # post-R2 gauge alignment (SKILL v2.2 §1): depth_body(p) ~ k * depth_anchor(p) + t over the object mask, no rotation;
            # depth_src = (depth_body - t) / k, and x, y scale with depth along the ray
            zb_body, mb = _proxy.render_perspective(V, F, K, *M_obj.shape)
            _, depth_anchor, pm_prov = A.anchor_pointmap(zh, float(dec["alpha"]), float(dec["b"]), K, valid, region=amask | M_obj)
            reg = mb & amask
            try:
                ga = D.gauge_align(zb_body, depth_anchor, reg, mode="median_ratio")
            except AssertionError as e:                                      # no overlap / non-positive scale: a recorded refusal on the experiment path
                if not ctx.get("auto"):
                    raise
                (out / "r2_refused.json").write_text(json.dumps({"schema": "m2e_r2_refused_v1", "variant": "body", "stage": "gauge_align", "reason": str(e),
                                                                   "n_region_px": int(reg.sum()), "body_iou_vs_anchor_mask": float(D.iou(mb, amask))}, indent=1), encoding="utf-8")
                print(f"[m2e] {it['clip']}/{it['category']}: Body gauge alignment refused: {e} -> r2_refused.json", flush=True); continue
            kk, tt, resid = ga["scale_k"], ga["shift_t"], ga["resid_rms_used"]
            zo = it["control"]["zeta_obj"].astype(np.float64); vis = M_obj & amask & (zo > 0)
            ga["median_depth_proxy_visible"] = float(np.median(1.0 / zo[vis])) if vis.any() else None
            ga["body_over_proxy_visible"] = (ga["median_depth_body"] / kk) / ga["median_depth_proxy_visible"] if ga["median_depth_proxy_visible"] else None
            V_src = D.apply_gauge(V, kk, tt)
            _, mb2 = _proxy.render_perspective(V_src, F, K, *M_obj.shape); iou_pre, iou_post = D.iou(mb, amask), D.iou(mb2, amask)
            assert abs(iou_post - iou_pre) < 1e-6, f"gauge changed the projection: IoU {iou_pre:.6f} -> {iou_post:.6f}"
            ga["projection_unchanged"] = {"iou_pre": iou_pre, "iou_post": iou_post}
            np.savez_compressed(out / "g_tar_mesh.npz", vertices=V_src.astype(np.float32), faces=F.astype(np.int32), K_1024=K.astype(np.float32), t_a=np.int64(it["t_a"]),
                                frame="camera_t_a", gauge="source (Body mesh depth aligned to band-alpha anchor depth: median-ratio scale about the camera centre, no rotation)")
            r2rec = {"schema": "m2e_r2_v1", "variant": "body", "pointmap": {"pointmap_source": "n/a (Body estimates its own depth)", "far_field_nonpositive_frac": pm_prov["far_field_nonpositive_frac"]}, "r2_seeds": [0], "r2_medoid_seed": 0,
                     "r2_iou_vs_anchor_mask": float(D.iou(mb, amask)), "gaussians_available": False,
                     "gauge_alignment": {**ga, "note": "keypoints kept in Body's own gauge for delta (a direction)"},
                     "alpha_used": dec["alpha"], "b_used": dec["b"], "alpha_source": dec["alpha_source"], "n_people": r.n_people, "kp_in_mask_frac": r.kp_in_mask_frac, "model_version": recon.model_version}
            (out / "r2_body.json").write_text(json.dumps({"kp3d_cam": kp.tolist()}, indent=0), encoding="utf-8")
            print(f"[m2e] {it['clip']}/{it['category']}: R2 body IoU vs anchor mask {r2rec['r2_iou_vs_anchor_mask']:.3f}; gauge k {kk:.4f} t {tt:+.4f} resid {resid:.4f} over {ga['n_pixels']} px")
        (out / "r2.json").write_text(json.dumps(r2rec, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    return 0


def run_qc(pairs: list, ctx: dict) -> int:
    recs = []
    for it in pairs:
        out = ctx["cache_m2e"] / it["clip"] / it["category"]
        if not (out / "r2.json").exists():
            why = json.loads((out / "r2_failed.json").read_text(encoding="utf-8"))["error"] if (out / "r2_failed.json").exists() else "r2.json missing (R2 not run or crashed)"
            print(f"[m2e] {it['clip']}/{it['category']}: NO R2 RECORD — {why}; excluded from this QC pass"); continue
        a = json.loads((out / "alpha.json").read_text(encoding="utf-8")); d = json.loads((out / "delta.json").read_text(encoding="utf-8")); r2 = json.loads((out / "r2.json").read_text(encoding="utf-8"))
        # reviewer 2026-09-27: what alpha PRODUCES — the object's depth in the source gauge per candidate — against C1's placement depth
        with np.load(out / "anchor_masks.npz") as zz:
            zobj, within = {}, {}
            for c in a["candidates"]:
                if c["alpha"] is None:
                    continue
                zh_c = zz[f"zeta_hat_seed_{c['seed']}"].astype(np.float64); m_c = zz[f"mask_seed_{c['seed']}"].astype(bool)
                disp = float(c["alpha"]) * zh_c + float(c["b"]); ok_c = m_c & (disp > 0)
                zobj[str(c["seed"])] = float(np.median(1.0 / disp[ok_c])) if ok_c.any() else None
                band_c, _ = A.ground_band(m_c, it["control"]["M_obj"].astype(bool), it["K_img"], np.asarray(it["stage0"]["n_plane"], float), it["control"]["valid"].astype(bool), it["hole"],
                                          int(ctx["cfg"]["alpha"]["band_inner_px"]), float(ctx["cfg"]["alpha"]["band_outer_frac"]))
                lo_b, hi_b = np.percentile(zh_c[band_c], [2, 98]); lo_o, hi_o = np.percentile(zh_c[m_c], [2, 98])
                within[str(c["seed"])] = {"object_zeta_hat_range": [float(lo_o), float(hi_o)], "band_zeta_hat_range": [float(lo_b), float(hi_b)], "inside": bool(lo_o >= lo_b and hi_o <= hi_b),
                                          "frac_object_inside_band_range": float(((zh_c[m_c] >= lo_b) & (zh_c[m_c] <= hi_b)).mean())}
        z_place = float(it["placement"]["placement"]["z"]); zs = [v for v in zobj.values() if v is not None]
        with np.load(out / "anchor_masks.npz") as zz:
            m_sel = zz[f"mask_seed_{a['selected_seed']}"].astype(bool)
        zo = it["control"]["zeta_obj"].astype(np.float64); vis = it["control"]["M_obj"].astype(bool) & m_sel & (zo > 0)
        z_proxy_vis = float(np.median(1.0 / zo[vis])) if vis.any() else None
        depth_rec = {"z_obj_per_candidate": zobj, "z_obj_selected": zobj.get(str(a["selected_seed"])), "z_placement_c1": z_place,
                     "z_obj_over_z_placement": (zobj.get(str(a["selected_seed"])) / z_place) if zobj.get(str(a["selected_seed"])) else None,
                     "z_proxy_visible_surface": z_proxy_vis, "z_obj_over_z_proxy_visible": (zobj.get(str(a["selected_seed"])) / z_proxy_vis) if (z_proxy_vis and zobj.get(str(a["selected_seed"]))) else None,
                     "z_obj_spread_rel": float(np.ptp(zs) / np.median(zs)) if len(zs) > 1 else None, "object_within_band_range": within.get(str(a["selected_seed"])),
                     "note": "alpha spread across seeds mixes the control's constraint with Depth-Anything's per-image gauge (reviewer 2026-09-27); z_obj is the quantity alpha produces"}
        rec = {"clip": it["clip"], "category": it["category"], "disposition": it["disposition"], "label": it["label"], "t_a": it["t_a"], "K_1024": it["K_img"].tolist(), "fx_over_fy": a["fx_over_fy"],
               "object_depth_source_gauge": depth_rec,
               "alpha": d["alpha_decision"]["alpha"], "b": d["alpha_decision"]["b"], "alpha_source": d["alpha_decision"]["alpha_source"], "flag": d["alpha_decision"].get("flag"),
               "alpha_band": a["candidates"][0]["band_fit"]["alpha"], "alpha_object_corrected": d["alpha_decision"].get("alpha_object_corrected"), "band_vs_object_ratio": d["alpha_decision"].get("band_vs_object_ratio"),
               "alpha_naive_over_reversed": a["candidates"][0]["band_fit"]["naive_over_reversed"], "alpha_boot_ci": a["candidates"][0]["band_bootstrap"],
               "alpha_per_candidate": a["alpha_per_candidate"], "alpha_spread_rel": a["alpha_spread_rel"], "alpha_low_confidence": a["candidates"][0]["alpha_low_confidence"],
               "delta_deg": d["delta_deg"], "delta_deg_abs": abs(d["delta_deg"]), "delta_instrument": d["delta_instrument"], "delta_definition": D.DELTA_DEFINITION,
               "delta_vs_g_tar_deg": d.get("delta_vs_g_tar_deg"), "front_back_margin": (d.get("scan") or {}).get("front_back_margin"), "g_front_back": d.get("g_front_back"),
               "gtar_shape_yaw": d.get("gtar_shape_yaw"), "flatten_predicted_yaw": d.get("flatten_predicted_yaw"), "r2_fallback": r2.get("fallback"),
               "delta_curve_width_deg": (d.get("scan") or {}).get("curve_width_deg"), "variant": r2["variant"], "r2_iou_vs_anchor_mask": r2["r2_iou_vs_anchor_mask"],
               "gaussians_available": r2["gaussians_available"], "pointmap_source": r2["pointmap"].get("pointmap_source"), "gauge_alignment": r2.get("gauge_alignment"),
               "r2_n_success": r2.get("r2_n_success"), "r2_per_seed": r2.get("per_seed"), "r2_object_depth": r2.get("object_depth"),
               "gate_outcomes": {**a["candidates"][0]["gates"], **d["gates"], "g_estimator_sign": a["candidates"][0]["g_estimator_sign"], "g_front_back": d.get("g_front_back"),
                                 "g_gauge_regression": {k_: (r2.get("gauge_alignment") or {}).get(k_) for k_ in ("k_regression", "k_regression_ci95", "k_regression_nonpositive", "anchor_depth_range_rel", "scale_k", "mode")} | {"outcome": "record"}},
               "config_hash": ctx["config_hash"],
               "upstream_hashes": {**it["upstream_hashes"], "alpha": cache.sha256_file(out / "alpha.json"), "delta": cache.sha256_file(out / "delta.json"), "r2": cache.sha256_file(out / "r2.json")}}
        # shape check (reviewer 2026-09-28): principal extents of G_tar vs the posed proxy (and vs the vendor-map reconstruction if the probe saved one)
        try:
            with np.load(out / "g_tar_mesh.npz") as z:
                Vg_s = z["vertices"].astype(np.float64)
            prox_s, _, _ = _proxy_mesh(ctx, it); pl_s = it["placement"]["placement"]
            V_pp = np.asarray(pl_s["X"], float) + float(pl_s["s"]) * _proxy.aligned_vertices(_proxy.ProxyMesh("proxy", prox_s["vertices"], prox_s["faces"], prox_s["vertices"].mean(0), prox_s["B_tar"]), np.asarray(pl_s["R_A"], float))
            shp = {"g_tar": D.principal_extents(Vg_s), "proxy_posed": D.principal_extents(V_pp)}
            vm = Path(ctx["drive"]) / "viz" / "m2e" / f"probe_vendor_mesh_{it['clip'][-8:]}_{it['category']}.npz"
            if vm.exists():
                with np.load(vm) as z:
                    shp["vendor_map_recon"] = D.principal_extents(z["vertices"].astype(np.float64))
            shp["g_tar_over_proxy_ratios"] = (np.array(shp["g_tar"]["ratios_to_largest"]) / np.maximum(np.array(shp["proxy_posed"]["ratios_to_largest"]), 1e-12)).tolist()
            rec["gtar_shape_extents"] = shp
        except Exception as e:                                                # noqa: BLE001
            rec["gtar_shape_extents"] = {"error": f"{type(e).__name__}: {e}"}
        Q.write_qc(out, rec); recs.append(rec)
        # review package for the HARD STOP: anchor with G_tar's silhouette (green) and the anchor mask (magenta), plus the delta curve
        try:
            import cv2
            img = cv2.imread(str(it["d_m2d"] / "candidates" / f"seed_{a['selected_seed']}.png"))
            with np.load(out / "g_tar_mesh.npz") as z:
                Vg, Fg = z["vertices"].astype(np.float64), z["faces"].astype(int)
            with np.load(out / "anchor_masks.npz") as z:
                am = z[f"mask_seed_{a['selected_seed']}"].astype(bool)
            _, mg = _proxy.render_perspective(Vg, Fg, it["K_img"], *am.shape)
            for m_, col in ((am, (255, 0, 255)), (mg, (0, 255, 0))):
                e = m_ ^ cv2.erode(m_.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool); img[e] = col
            pane = np.full((img.shape[0], 512, 3), 30, np.uint8)
            if d.get("scan"):
                cur = np.array(d["scan"]["curve"]); xs = ((cur[:, 0] + d["scan"]["scan_deg"]) / (2 * d["scan"]["scan_deg"]) * 480 + 16).astype(int)
                ys = (img.shape[0] - 40 - cur[:, 1] * (img.shape[0] - 80)).astype(int)
                for i in range(len(xs) - 1):
                    cv2.line(pane, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]), (0, 200, 255), 2)
                cv2.putText(pane, f"delta {d['delta_deg']:+.1f} deg  width {d['scan']['curve_width_deg']:.0f}", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            else:
                cv2.putText(pane, f"delta {d['delta_deg']:+.1f} deg (keypoints)", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(img, f"{it['clip'][-8:]} {it['category']} ({it['disposition']})  alpha {rec['alpha']:.3f} ({rec['alpha_source']})  G_tar IoU {rec['r2_iou_vs_anchor_mask']:.3f}", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            if rec.get("r2_fallback"):                                        # hard-stop verdict 3: vendor placement vs a CONTACT-PLANE re-placement, side by side (render only; the cache is untouched)
                # Reviewer 2026-09-28: not a slide in the image plane at constant depth (that lifts or sinks the car in 3-D) but C1's own
                # placement solve with G_tar in place of the proxy: rotation (G_tar's) and scale (the gauge k) fixed, translation from
                # (a) the mesh centroid on the ray through the anchor-mask centroid and (b) ground contact n.X + d = 0 at the lowest vertex.
                ja, ia = np.nonzero(am)
                if len(ja):
                    Kk = it["K_img"]; n_pl = np.asarray(it["stage0"]["n_plane"], float); n_pl /= np.linalg.norm(n_pl)
                    # the plane offset C1 actually placed against: the LOCAL contact-band offset when one was observed (284: d_used 0.369 vs
                    # global 0.230 — the clip whose road camber differs from the global plane), else the global d_plane
                    d_used = it["placement"]["placement"].get("d_used"); d_pl = float(d_used) if d_used is not None and np.isfinite(float(d_used)) else float(it["stage0"]["d_plane"])
                    ray = np.linalg.solve(Kk, np.array([ia.mean() + 0.5, ja.mean() + 0.5, 1.0])); ray /= np.linalg.norm(ray)
                    c_mesh = Vg.mean(0); rel = Vg - c_mesh
                    lam = -(d_pl + float((rel @ n_pl).min())) / float(ray @ n_pl)          # lowest vertex on the plane: min_v n.(rel_v + lam ray) + d = 0
                    t_vec = lam * ray - c_mesh; Vc = Vg + t_vec
                    _, mc = _proxy.render_perspective(Vc, Fg, Kk, *am.shape)
                    img2 = cv2.imread(str(it["d_m2d"] / "candidates" / f"seed_{a['selected_seed']}.png"))
                    for m_, col in ((am, (255, 0, 255)), (mc, (0, 255, 0))):
                        e = m_ ^ cv2.erode(m_.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool); img2[e] = col
                    iou_c = float((mc & am).sum() / max((mc | am).sum(), 1))
                    z_c, z_v = float(np.median(Vc[:, 2])), float(np.median(Vg[:, 2]))
                    cv2.putText(img2, f"contact-plane re-placement: IoU {iou_c:.3f} (vendor {rec['r2_iou_vs_anchor_mask']:.3f}); depth {z_c:.3f} vs gauge {z_v:.3f}", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                    img = np.concatenate([img, img2], 1)
                    rec["contact_plane_replacement"] = {"translation": t_vec.tolist(), "lambda_along_ray": float(lam), "iou_vs_anchor_mask": iou_c, "median_depth_replaced": z_c, "median_depth_gauge": z_v,
                                                        "plane_offset_used": d_pl, "plane_offset_global": float(it["stage0"]["d_plane"]), "z_placement_c1": float(it["placement"]["placement"]["z"]),
                                                        "depth_ratio_replaced_over_gauge": z_c / z_v, "passes_m3_footprint_gate_0p7": bool(iou_c >= 0.7),
                                                        "note": "render + numbers only; not applied to G_tar (hard-stop verdict 3); rotation = G_tar's, scale = gauge k"}
            viz0 = Path(ctx["drive"]) / "viz" / "m2e"; viz0.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(viz0 / f"{it['clip']}_{it['category']}_gtar.png"), np.concatenate([img, pane], 1))
        except Exception as e:                                                # noqa: BLE001 — the review PNG must not block the record
            print(f"[m2e] review PNG failed for {it['clip']}/{it['category']}: {type(e).__name__}: {e}")
        print(f"[m2e] {it['clip']}/{it['category']} ({rec['disposition']}): alpha {rec['alpha']:.4f} ({rec['alpha_source']}), delta {rec['delta_deg']:+.2f} deg ({rec['delta_instrument']}), R2 IoU {rec['r2_iou_vs_anchor_mask']:.3f}")
    viz = Path(ctx["drive"]) / "viz" / "m2e"; viz.mkdir(parents=True, exist_ok=True)
    summ = Q.accepted_summary(recs)
    L = ["# m2_e — QC summary", "", "| clip | cat | disp. | alpha | source | band/obj (corr.) | delta (deg) | instrument | width | front/back margin | g_front_back | vs G_tar (scan) | G_tar shape yaw | flatten-pred. yaw | R2 IoU (n ok) | fallback | z_obj / z_C1 | z_obj / z_proxy_vis | z_obj spread | obj in band range | flag |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        fb = r.get("g_front_back") or {}; dz = r["object_depth_source_gauge"]; wb = dz.get("object_within_band_range") or {}
        L.append(f"| {r['clip'][-8:]} | {r['category']} | {r['disposition']} | {r['alpha']:.4f} | {r['alpha_source']} | {r['band_vs_object_ratio'] if r['band_vs_object_ratio'] is None else round(r['band_vs_object_ratio'], 3)} | "
                 f"{r['delta_deg']:+.2f} | {r['delta_instrument']} | {r['delta_curve_width_deg']} | {r['front_back_margin'] if r['front_back_margin'] is None else round(r['front_back_margin'], 3)} | {fb.get('outcome')} | "
                 f"{None if r['delta_vs_g_tar_deg'] is None else round(r['delta_vs_g_tar_deg'], 1)} | {None if not r.get('gtar_shape_yaw') else round(r['gtar_shape_yaw']['delta_gtar_shape_deg'], 1)} | "
                 f"{None if not r.get('flatten_predicted_yaw') else round(r['flatten_predicted_yaw']['predicted_signed_deg'], 1)} | "
                 f"{r['r2_iou_vs_anchor_mask']:.3f} ({r.get('r2_n_success', '-')}) | {r.get('r2_fallback') or ''} | "
                 f"{None if dz['z_obj_over_z_placement'] is None else round(dz['z_obj_over_z_placement'], 3)} | {None if dz.get('z_obj_over_z_proxy_visible') is None else round(dz['z_obj_over_z_proxy_visible'], 3)} | {None if dz['z_obj_spread_rel'] is None else round(dz['z_obj_spread_rel'], 3)} | "
                 f"{wb.get('inside')} ({round(wb.get('frac_object_inside_band_range', 0), 2)}) | {r['flag'] or ''} |")
    L += ["", D.DELTA_DEFINITION + ". Read |delta| against the curve width, not as a point estimate (the scan resolves ~+-2 deg at best).",
          "front/back margin = peak IoU(posed) - peak IoU(proxy flipped 180): small = ambiguous pair (a flipped anchor scores a plausible quiet delta; test 5)."]
    L += ["", "Accepted-set summary (labelled pairs excluded): " + json.dumps(summ), "", "HARD STOP (SKILL §11): human review of G_tar renders and the delta curves before Module 3."]
    (viz / "report_m2e_qc.md").write_text(chr(10).join(L) + chr(10), encoding="utf-8")
    for r in recs:
        (viz / f"qc_m2e_{r['clip'][-8:]}_{r['category']}.json").write_text(json.dumps({"schema": Q.QC_SCHEMA, **r}, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    print(f"[m2e] -> {viz / 'report_m2e_qc.md'}")
    return 0


def run_handoff(pairs: list, ctx: dict) -> int:
    """Per-pair handoff.json for Module 3 (docs/contract_m2e_to_m3.md §1). Written only after the hard stop: `placement_variant`
    for fallback pairs and the per-pair acceptance come from configs/m2e.yaml: handoff (the human's verdicts), never defaulted."""
    hv = ctx["cfg"].get("handoff") or {}
    auto = bool(ctx.get("auto"))
    assert auto or hv.get("hard_stop_declared") is True, "handoff refused: the hard stop has not been declared in configs/m2e.yaml: handoff.hard_stop_declared"
    for it in pairs:
        key = f"{it['clip']}/{it['category']}"; out = ctx["cache_m2e"] / it["clip"] / it["category"]
        q = json.loads((out / "qc_m2e.json").read_text(encoding="utf-8"))
        if auto and it["disposition"] == "experiment":
            # the experiment path: the verdict is the gate outcomes alone (every gate pass / ungated / n/a / unchecked; any 'fail' refuses), recorded as such
            go = q.get("gate_outcomes") or {}
            failed = sorted(k for k, v in go.items() if isinstance(v, dict) and v.get("outcome") == "fail")
            verdict = "refused" if failed else "pass"; verdict_source = {"source": "automatic: gate outcomes of qc_m2e.json", "failed_gates": failed, "unchecked": sorted(k for k, v in go.items() if isinstance(v, dict) and v.get("outcome") == "unchecked")}
            if failed:
                (out / "handoff_refused.json").write_text(json.dumps({"pair": key, **verdict_source}, indent=1), encoding="utf-8")
                print(f"[m2e] {key}: refused by its gates {failed}; no handoff written (handoff_refused.json)"); continue
        else:
            verdict = (hv.get("pairs") or {}).get(key); verdict_source = {"source": "configs/m2e.yaml handoff.pairs (the human's verdict)"}
            assert verdict in ("pass", "refused"), f"{key}: no hard-stop verdict in configs/m2e.yaml: handoff.pairs (pass | refused)"
            if verdict == "refused":
                print(f"[m2e] {key}: refused at the hard stop; no handoff written"); continue
        a = json.loads((out / "alpha.json").read_text(encoding="utf-8"))
        r2 = json.loads((out / "r2.json").read_text(encoding="utf-8")); st0 = it["stage0"]; pl = it["placement"]["placement"]
        F = int(st0["qc"].get("n_frames") or 0)
        if not F:
            from src.preprocess.stage_c import finalize
            M, present, _ = finalize.read_mask(Path(ctx["cfg_a"]["paths"]["stage_c_cache"]) / it["clip"]); F = int(len(present))
        pv = None
        if r2.get("fallback"):
            pv = (hv.get("placement_variant") or {}).get(key)
            if auto and it["disposition"] == "experiment" and pv is None:
                (out / "handoff_refused.json").write_text(json.dumps({"pair": key, "source": "automatic", "reason": "fallback reconstruction: the placement variant is a human decision (handoff.placement_variant); refused on the experiment path"}, indent=1), encoding="utf-8")
                print(f"[m2e] {key}: fallback reconstruction needs a human placement variant; refused on the experiment path (handoff_refused.json)"); continue
            assert pv in ("vendor", "contact_plane"), f"{key}: fallback pair needs handoff.placement_variant (vendor | contact_plane)"
        gauge = (r2.get("gauge_alignment") or {}).get("scale_k")
        h = {"schema": "m2e_handoff_v1", "clip": it["clip"], "category": it["category"], "disposition": it["disposition"], "label": it["label"],
             "anchor_image": str(it["d_m2d"] / "candidates" / f"seed_{a['selected_seed']}.png"), "anchor_pixels": "non-square (fx/fy below)",
             "K_1024": it["K_img"].tolist(), "K_a": st0["K_a"].astype(float).tolist(), "fx_over_fy": float(it["K_img"][0, 0] / it["K_img"][1, 1]),
             "t_a": it["t_a"], "n_frames_source": F, "n_frames_after_t_a": F - it["t_a"], "generated_frame0_is_source_frame": it["t_a"],
             "g_tar_mesh": str(out / "g_tar_mesh.npz"), "g_tar_gauss": str(out / "g_tar_gauss.npz") if (out / "g_tar_gauss.npz").exists() else None,
             "gaussians_available": bool((out / "g_tar_gauss.npz").exists()), "pointmap_source": r2["pointmap"].get("pointmap_source"), "placement_variant": pv,
             "A_deg": q["delta_deg"], "A_instrument": q["delta_instrument"], "A_definition": q.get("delta_definition"), "A_already_applied_in_g_tar": True,
             "alpha": q["alpha"], "b": q["b"], "alpha_ci_block32": a["candidates"][0]["band_bootstrap"]["block_32"]["ci"], "alpha_source": q["alpha_source"], "gauge_k": gauge,
             "placement_C1": {"X": np.asarray(pl["X"], float).tolist(), "s": float(pl["s"]), "R_A": np.asarray(pl["R_A"], float).tolist(), "z": float(pl["z"])},
             "n_plane": st0["n_plane"].astype(float).tolist(), "d_plane": float(st0["d_plane"]),
             "d_used": float(pl["d_used"]) if pl.get("d_used") is not None and np.isfinite(float(pl["d_used"])) else float(st0["d_plane"]),
             "contact_plane_note": "every contact computation uses d_used (C1's local contact-band offset when observed), never d_plane (contract v1 / v2.1 amendment)",
             "d_src": st0["d_src"].astype(float).tolist(), "frame": "camera_t_a", "gauge": "source (D4RT)",
             "R2_iou": q["r2_iou_vs_anchor_mask"], "gtar_shape_yaw_deg": (q.get("gtar_shape_yaw") or {}).get("delta_gtar_shape_deg"), "gtar_shape_extents": q.get("gtar_shape_extents"),
             "known_properties": {"heading_accuracy_ungated": True, "within_object_scale_error": "7-27 % in the point map; does NOT reach G_tar's shape (extents match the vendor-map reconstruction)",
                                  "r2_orientation_error": hv.get("known_r2_orientation_error_deg"), "human_facing_offset_deg": (__import__("src.anchor.subjects", fromlist=["facing_offset"]).facing_offset(it["category"]) or None),
                                  "contact_plane_replacement": q.get("contact_plane_replacement") if pv == "contact_plane" else None},
             "upstream_hashes": {**q["upstream_hashes"], "qc_m2e": cache.sha256_file(out / "qc_m2e.json")}, "hard_stop": {"verdict": verdict, "declared": (None if verdict_source["source"].startswith("automatic") else hv.get("declared_on")), **verdict_source}}
        (out / "handoff.json").write_text(json.dumps(h, indent=1), encoding="utf-8")
        print(f"[m2e] {key}: handoff written (t_a {it['t_a']}, {F - it['t_a']} source frames after t_a, gaussians {h['gaussians_available']}, placement {pv})")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", action="store_true"); ap.add_argument("--delta", action="store_true"); ap.add_argument("--r2", action="store_true")
    ap.add_argument("--qc", action="store_true"); ap.add_argument("--handoff", action="store_true"); ap.add_argument("--all", action="store_true")
    ap.add_argument("--clip"); ap.add_argument("--category"); ap.add_argument("--config", default="configs/stage_a.yaml")
    ap.add_argument("--auto", action="store_true", help="the experiment path: pairs without a human disposition run as 'experiment'; the handoff verdict is the gate outcomes (plan v1)")
    args = ap.parse_args()
    ctx = load_ctx(args); ctx["auto"] = bool(args.auto)
    disp = ctx["cfg"]["dispositions"]
    todo = [tuple(k.split("/")) for k in list(disp["accepted"]) + list(disp["labelled"])] if args.all else [(args.clip, args.category)]
    pairs, skipped = [], []
    for clip, cat in todo:
        try:
            pairs.append(pair_inputs(clip, cat, ctx))
        except (AssertionError, FileNotFoundError) as e:
            if not args.all:
                raise
            skipped.append((clip, cat, str(e).splitlines()[0][:140])); print(f"[m2e] {clip}/{cat}: SKIPPED ({skipped[-1][2]})")
    assert pairs, "no pair to run"
    print(f"[m2e] {len(pairs)} pair(s): " + ", ".join(f"{p['clip'][-8:]}/{p['category']} ({p['disposition']})" for p in pairs))
    if args.alpha:
        return run_alpha(pairs, ctx)
    if args.delta:
        return run_delta(pairs, ctx)
    if args.r2:
        return run_r2(pairs, ctx)
    if args.qc:
        return run_qc(pairs, ctx)
    if args.handoff:
        return run_handoff(pairs, ctx)
    print("[m2e] nothing requested (--alpha | --delta | --r2 | --qc)"); return 0


if __name__ == "__main__":
    sys.exit(main())
