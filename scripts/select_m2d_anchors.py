#!/usr/bin/env python
"""Module 2 anchor selection v2, the runner (reviewer's rule a' with the Q1/Q2 answers of 2026-10-17; definitions in docs/plan_experiment_v1.md).

  measure   per candidate anchor image (cache/m2_d/<clip>/<cat>/candidates/seed_<k>.png):
              G1  the Module 3 preconditions on the candidate: MoGe depth under the pair's K_1024, k_bg on the ground band with a block
                  bootstrap (band points >= lock m3.selection.band_points_min, half-split and bootstrap half-width <= lock), and the contact
                  gate s_c read against the PROXY's depth (control.npz zeta_obj on M_obj: the placement every candidate was generated on),
                  |r_c - 1| <= lock m3.int.contact_tol;
              G2  path support f_sup: the source object's C.5 path carried by the census/record camera into camera-t_a, projected through
                  K_1024 onto the candidate, the lowest projected object point per in-view frame tested against the candidate's ground-plane
                  inliers (contact.fit_ground_plane on the candidate's depth); gated only once lock m2.selection.tau_sup exists (after its
                  self-test), reported before;
              H1  silhouette margin m_H = IoU(mask, P) - IoU(mask, flip_x P), P = M_obj; >= delta_h pass, <= -delta_h fail, else -> H2
                  (H2 = "heading unchecked" until the facing instrument passes its per-class self-test; recorded, never assumed);
              quality floor MUSIQ >= tau_q; proxy IoU; CLIP identity; object found (the segment instrument of record).
            -> cache/m2_d/<clip>/<cat>/selection_v2/candidates.json (append-only per candidate; idempotent on resume)
  select    the rule (src/anchor/m2d_select_v2.py) on the measured candidates -> selection_v2/selection.json per pair, the comparison with
            the human's selection and with the blind verdicts (measurements/m2d_anchors/blind_passfail_template.csv), the sensitivity
            (eps halved / doubled), and reports/m2_selection/selection.{json,md}. CPU.

Selection uses Module 2's outputs only (the candidate image, the control, the camera of record): never a Module 3 video.

    python scripts/select_m2d_anchors.py measure --pairs 000000000265.2.005/human ...      # GPU (MoGe, OWLv2, SAM2, CLIP, MUSIQ)
    python scripts/select_m2d_anchors.py select  --pairs ...                                # CPU
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.anchor.m2d import segment as SEG                                        # noqa: E402
from src.anchor.m2d_select_v2 import Candidate, select as rule_select            # noqa: E402
from src.anchor.m2e import alpha as m2e_alpha                                     # noqa: E402
from src.generate.m3 import contact, depth_moge, handoff as H                     # noqa: E402
from src.preprocess.stage_c import finalize                                       # noqa: E402

CLIP_ID = "openai/clip-vit-large-patch14"
CLIP_TEXT = {"vehicle": "a red car", "human": "a man in a grey jacket and dark trousers"}
FOCAL_REL_TOL = 0.01


def iou(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool); u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def flip_about_centroid(P):
    ys, xs = np.nonzero(P)
    if not len(xs):
        return P.copy()
    out = np.zeros_like(P); out[ys, np.clip(np.round(2 * xs.mean() - xs).astype(int), 0, P.shape[1] - 1)] = True
    return out


def kbg_fit(inv_zeta, D0, region, block_px, n_boot, seed=0):
    """k_bg = median over the band of (1/zeta_bg)/D0; block bootstrap; half-split by column parity (the lock's determinability tests)."""
    r = inv_zeta / D0; ok = region & np.isfinite(r) & (r > 0)
    n = int(ok.sum())
    if n == 0:
        return {"n_band_points": 0}
    k = float(np.median(r[ok])); vv, uu = np.nonzero(ok)
    blk = (vv // block_px) * 100000 + (uu // block_px); ids = np.unique(blk); by = {b: r[vv[blk == b], uu[blk == b]] for b in ids}
    rng = np.random.default_rng(seed)
    boots = [float(np.median(np.concatenate([by[b] for b in rng.choice(ids, len(ids))]))) for _ in range(n_boot)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    left, right = np.median(r[ok & ((np.indices(r.shape)[1] // block_px) % 2 == 0)]), np.median(r[ok & ((np.indices(r.shape)[1] // block_px) % 2 == 1)])
    return {"k_bg": k, "ci95": [float(lo), float(hi)], "bootstrap_halfwidth_rel": float((hi - lo) / 2 / k), "half_split_rel": float(abs(left - right) / k),
            "n_band_points": n, "n_blocks": int(len(ids))}


def ground_inliers(depth, valid, obj_mask, Kq, cc):
    pl = contact.fit_ground_plane(depth, valid, obj_mask, Kq, int(cc["plane_nbhd_px"]), int(cc["ransac_iters"]), float(cc["ransac_rel_tol"]), int(cc["min_inliers"]), int(cc["seed"]))
    if pl is None:
        return None, None
    Hh, Ww = depth.shape; vv, uu = np.nonzero(valid & np.isfinite(depth) & (depth > 0))
    X = (np.linalg.inv(Kq) @ np.stack([uu + 0.5, vv + 0.5, np.ones(len(uu))], 0)).T * depth[vv, uu][:, None]
    inl = np.zeros((Hh, Ww), bool); inl[vv, uu] = np.abs(X @ pl["n"] + pl["d"]) / X[:, 2] < float(cc["ransac_rel_tol"])
    return pl, inl


def object_path_world(root, clip, t_a, F, inverse=False):
    """the source object's C.5 dense points per frame in camera-t_a world coordinates, through the camera of record (camera C):
    camera_c.npz E_c[k] is world -> camera-(t_a + k), indexed by k from t_a (scripts/check_m3_camera_c.py: lifting uses C_k = inv(E_c[k]),
    reprojection uses reproj_px(X_w, E_c[k], K[t])). `inverse` applies E_c[k] the wrong way round: the convention-check fixture only."""
    with np.load(root / "cache" / "stage_c5" / f"{clip}.npz") as z:
        Q = z["Q_dense"].astype(bool); X = z["X_local_dense"].astype(np.float64)          # the same-frame C.5 lift (camera-t coordinates), as the record's object path uses it
    E_c, _ = load_camera_of_record(root, clip, t_a)
    out = []
    for k in range(F):
        t = t_a + k
        if t >= X.shape[1] or k >= len(E_c) or not np.isfinite(E_c[k]).all():
            out.append(None); continue
        vq = Q[:, t] & np.isfinite(X[:, t]).all(1)
        if vq.sum() < 10:
            out.append(None); continue
        Xc = X[vq, t]; C = E_c[k] if inverse else np.linalg.inv(E_c[k])
        out.append(Xc @ C[:3, :3].T + C[:3, 3])
    return out, None


def load_camera_of_record(root, clip, t_a):
    """camera C + BA (cache/m3/<clip>/camera_c_ba.npz, the gate of record's camera) when present, else camera C; E[k] world -> camera-(t_a+k)."""
    d = root / "cache" / "m3" / clip
    f = d / "camera_c_ba.npz" if (d / "camera_c_ba.npz").exists() else d / "camera_c.npz"
    cc = np.load(f); E = cc["E_c"].astype(np.float64)
    if "t_a" in cc.files:
        assert int(cc["t_a"]) == t_a, f"{f.name} t_a {int(cc['t_a'])} != {t_a}"
    assert np.allclose(E[0], np.eye(4)), f"{f.name}: E[0] is not the identity (world = camera-t_a)"
    return E, f.name


def static_track_fixture(root, clip, t_a, F, inverse=False):
    """Rule 17 fixture for the composition: STATIC background tracks (Stage A tracks with Stage B y_dyn False) carried from frame t to
    the world through G2's transform and projected into frame t_a must land on their own observed pixel at t_a. Median px over frames;
    comparable to the camera gate's held-out error (lock m3.trajectory.camera_c_gate_patch in patches)."""
    with np.load(root / "cache" / "stage_a" / clip / "bundle.npz") as z:
        K, Xl, P, Q = z["K"].astype(np.float64), z["X_local"].astype(np.float64), z["P"].astype(np.float64), z["Q"].astype(bool)
    with np.load(root / "cache" / "stage_b" / f"{clip}.npz") as z:
        static = ~z["y_dyn"].astype(bool)
    M, _, _ = finalize.read_mask(root / "cache" / "stage_c" / clip)
    def on_subject(t):
        ok = Q[:, t]; u = P[:, t, 0].astype(int).clip(0, M.shape[2] - 1); v = P[:, t, 1].astype(int).clip(0, M.shape[1] - 1)
        return ok & M[t].astype(bool)[v, u]
    static &= ~on_subject(t_a)
    E_c, _ = load_camera_of_record(root, clip, t_a)
    errs = []
    for k in range(1, F):
        t = t_a + k
        if t >= Xl.shape[1] or k >= len(E_c) or not np.isfinite(E_c[k]).all():
            continue
        v = static & Q[:, t] & Q[:, t_a] & ~on_subject(t)
        if v.sum() < 20:
            continue
        C = E_c[k] if inverse else np.linalg.inv(E_c[k])
        Xw = Xl[v, t] @ C[:3, :3].T + C[:3, 3]
        uvw = (K[t_a] @ Xw.T).T; ok = uvw[:, 2] > 0
        if ok.sum() < 20:
            continue
        pix = uvw[ok, :2] / uvw[ok, 2:3]
        errs.append(float(np.median(np.linalg.norm(pix - P[v, t_a][ok], axis=1))))
    return (float(np.median(errs)) if errs else None), len(errs)


def contact_hits(path, K, plane, inl, Hh, Ww, rel_tol, band_px):
    """G2's hit test, geometric (reviewer 2026-10-19): the lowest 5 % of the projected object points per in-view frame must lie on the
    ground plane in 3-D (|n.X + d| / Z <= rel_tol) AND project within band_px of a ground inlier pixel (distance transform of the inlier
    mask). Returns (f_sup, n_frames_in_view, per_frame)."""
    import cv2
    dist = cv2.distanceTransform((~inl).astype(np.uint8), cv2.DIST_L2, 3) if inl is not None else None
    hits = n_in = 0; per = []
    for Xw in path:
        if Xw is None:
            per.append(None); continue
        uvw = (K @ Xw.T).T; z_ = uvw[:, 2]; okz = z_ > 0
        if okz.sum() < 10:
            per.append(None); continue
        Xo = Xw[okz]; u, v = uvw[okz, 0] / z_[okz], uvw[okz, 1] / z_[okz]
        inview = (u >= 0) & (u < Ww) & (v >= 0) & (v < Hh)
        if inview.sum() < 10:
            per.append(None); continue
        n_in += 1
        lowest = np.argsort(v[inview])[-max(5, int(0.05 * inview.sum())):]
        Xc = Xo[inview][lowest]; uc = u[inview][lowest].astype(int).clip(0, Ww - 1); vc = v[inview][lowest].astype(int).clip(0, Hh - 1)
        on_plane = np.abs(Xc @ plane["n"] + plane["d"]) / Xc[:, 2] <= rel_tol
        near = dist[vc, uc] <= band_px if dist is not None else np.zeros(len(uc), bool)
        h = int((on_plane & near).mean() >= 0.5); hits += h; per.append(h)
    return ((hits / n_in) if n_in else None), n_in, per


def measure(args):
    import cv2, torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    from pyiqa.archs.musiq_arch import MUSIQ
    from huggingface_hub import hf_hub_download
    from moge.model.v2 import MoGeModel
    lock_all = yaml.safe_load((ROOT / "configs" / "calibration.lock.yaml").read_text(encoding="utf-8")); lock = lock_all["m3"]; sel = lock_all["m2"]["selection"]
    m3cfg = yaml.safe_load((ROOT / "configs" / "m3.yaml").read_text(encoding="utf-8"))["m3"]; cc = m3cfg["contact"]
    acfg = yaml.safe_load((ROOT / "configs" / "m2e.yaml").read_text(encoding="utf-8"))["m2e"]["alpha"]
    root = Path(args.drive_root); dev = "cuda"
    model = MoGeModel.from_pretrained(args.checkpoint).to(dev).eval(); seg = SEG.load_segmenter()
    clip_p = CLIPProcessor.from_pretrained(CLIP_ID); clip_m = CLIPModel.from_pretrained(CLIP_ID).to(dev).eval()
    musiq = MUSIQ(pretrained_model_path=str(hf_hub_download("chaofengc/IQA-PyTorch-Weights", "musiq_koniq_ckpt-e95806b9.pth")), num_class=1).to(dev).eval()
    lat = lock["lattice"]; px = int(lat["px_per_patch"]); Wn, Hn = int(lat["Wp"]) * px, int(lat["Hp"]) * px
    for pair in args.pairs:
        clip, cat = pair.split("/"); d2 = root / "cache" / "m2_d" / clip / cat; out = d2 / "selection_v2"; out.mkdir(exist_ok=True)
        jf = out / "candidates.json"; recs = json.loads(jf.read_text(encoding="utf-8")) if jf.exists() else []
        done = {r["seed"] for r in recs}
        hp = root / "cache" / "m2_e" / clip / cat / "handoff.json"; hc = root / "cache" / "census" / clip / cat / "handoff.json"
        h = json.loads((hp if hp.exists() else hc).read_text(encoding="utf-8")); handoff_source = "m2_e" if hp.exists() else "census (source-side)"
        K1024 = np.asarray(h["K_1024"], np.float64); t_a = int(h["t_a"]); n_plane = np.asarray(h["n_plane"], float)
        with np.load(root / "cache" / "m2_c" / clip / cat / "control.npz") as z:
            M_obj, valid_c, zeta_ctrl, zeta_obj = z["M_obj"].astype(bool), z["valid"].astype(bool), z["zeta_ctrl"].astype(np.float64), z["zeta_obj"].astype(np.float64)
        M, _, _ = finalize.read_mask(root / "cache" / "stage_c" / clip)
        hole = cv2.resize(M[t_a].astype(np.uint8), (1024, 1024), interpolation=cv2.INTER_NEAREST).astype(bool)
        Pf = flip_about_centroid(M_obj)
        path, path_err = object_path_world(root, clip, t_a, int(args.frames))
        for k in range(int(sel.get("n_seeds", 4))):
            if k in done and not args.force:
                continue
            recs = [r for r in recs if r["seed"] != k]
            ip = d2 / "candidates" / f"seed_{k}.png"
            if not ip.exists():
                recs.append({"seed": k, "object_found": False, "reason": "candidate image missing"}); continue
            bgr = cv2.imread(str(ip)); img = bgr[..., ::-1].copy()
            rec = {"seed": k, "pair": pair, "handoff_source": handoff_source}
            # --- the subject mask (instrument of record), proxy IoU, silhouette margin, CLIP, MUSIQ
            mask, info = SEG.segment(seg, img, cat); rec["object_found"] = bool(mask.any()); rec["segment"] = info
            if rec["object_found"]:
                rec["object_iou"] = iou(mask, M_obj); rec["m_H"] = iou(mask, M_obj) - iou(mask, Pf)
                ys, xs = np.nonzero(mask); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max(); w, hgt = x1 - x0, y1 - y0
                cr = img[max(0, int(y0 - .2 * hgt)):min(1024, int(y1 + .2 * hgt)), max(0, int(x0 - .2 * w)):min(1024, int(x1 + .2 * w))]
                inputs = clip_p(text=[CLIP_TEXT[cat]], images=[Image.fromarray(cr)], padding=True, return_tensors="pt")
                with torch.inference_mode():
                    r_ = clip_m(**{kk: v.to(dev) for kk, v in inputs.items()})
                rec["subject_similarity"] = float((r_.image_embeds.float() @ r_.text_embeds.float().T)[0, 0].cpu())
            x = torch.from_numpy(np.asarray(img, np.float32) / 255).permute(2, 0, 1)[None].to(dev)
            with torch.inference_mode():
                rec["quality_full"] = float(musiq(x).item())
            # --- G1: depth, band determinability, contact against the proxy
            amask = mask if rec["object_found"] else M_obj
            band, bprov = m2e_alpha.ground_band(amask, M_obj, K1024, n_plane, valid_c, hole, int(acfg["band_inner_px"]), float(acfg["band_outer_frac"]))
            dep = depth_moge.estimate_depth(model, img, K1024, Wn, Hn, FOCAL_REL_TOL, dev)
            Wq, Hq, Kq, D0, mk = dep["W_sq"], dep["H"], dep["K_sq"], dep["depth"], dep["mask"]
            rs = lambda m_: cv2.resize(m_.astype(np.float32), (Wq, Hq), interpolation=cv2.INTER_AREA)            # noqa: E731
            band_q = rs(band) > 0.999; amask_q = rs(amask) > 0.5; Mobj_q = rs(M_obj) > 0.5
            with np.errstate(divide="ignore"):
                inv_zeta = np.where(zeta_ctrl > 0, 1.0 / zeta_ctrl, np.nan)
            inv_q = cv2.resize(inv_zeta.astype(np.float32), (Wq, Hq), interpolation=cv2.INTER_AREA).astype(np.float64)
            kb = kbg_fit(inv_q, D0, band_q & mk, int(lock["int"]["kbg_block_px"]), int(args.n_boot))
            g1 = {"band": kb, "band_points_min": int(lock["selection"]["band_points_min"]), "kbg_ci_rel_width": (None if "ci95" not in kb else float((kb["ci95"][1] - kb["ci95"][0]) / kb["k_bg"])),
                  "kbg_ci_gate": lock["int"].get("kbg_ci_rel_max"), "definition": "the record's admission as applied (reviewer 2026-10-18): band points >= min; k_bg CI width recorded, gated only by lock m3.int.kbg_ci_rel_max (null = ungated); contact outcome in the record's admitted set"}
            g1["band_determinable"] = bool(kb.get("n_band_points", 0) >= g1["band_points_min"]) and (g1["kbg_ci_gate"] is None or (g1["kbg_ci_rel_width"] is not None and g1["kbg_ci_rel_width"] <= float(g1["kbg_ci_gate"])))
            if g1["band_determinable"]:
                Dsrc = kb["k_bg"] * D0
                with np.errstate(divide="ignore"):
                    zbuf = np.where(M_obj & (zeta_obj > 0), 1.0 / zeta_obj, np.nan)                       # the proxy's depth as placed (the control's object field)
                zbuf_q = cv2.resize(zbuf.astype(np.float32), (Wq, Hq), interpolation=cv2.INTER_NEAREST).astype(np.float64)
                rc = contact.contact_ratio(zbuf_q, Mobj_q, Dsrc, mk, amask_q, float(lock["int"]["contact_rows"]), int(lock["int"]["contact_band_px"]), int(lock["int"]["kbg_block_px"]), int(args.n_boot), 0)
                pl, inl = ground_inliers(Dsrc, mk, amask_q | Mobj_q, Kq, cc)
                if pl is not None:
                    rc.update(contact.contact_ratio_plane(zbuf_q, Mobj_q, pl, Kq, float(lock["int"]["contact_rows"]), int(lock["int"]["kbg_block_px"]), int(args.n_boot), 0, anchor_mask=amask_q))
                rc["object_reaches_frame_bottom"] = bool(amask_q[-2:].any())
                dec = contact.decision(rc, float(lock["int"]["contact_tol"]), px)
                g1["contact"] = {"r_c_B": dec.get("r_c_B"), "outcome": dec["outcome"], "s_c": dec["s_c"], "n_ground_px": rc.get("n_ground_px"), "instrument": "the proxy as placed (control zeta_obj), not the reconstructed mesh of the record"}
                g1["contact_within_tol"] = dec["outcome"] in ("contact_holds", "no_contact_line_in_view", "contact_fails_regauge")   # the record's admitted outcomes; label reported
                # --- G2: path support
                if path is not None and inl is not None:
                    # the candidate is on the square grid (Wq x Hq) under Kq: the object points (camera-t_a world) project with Kq
                    f_, n_in, per_ = contact_hits(path, Kq, pl, inl, Hq, Wq, float(cc["ransac_rel_tol"]), int(lock["int"]["contact_band_px"]))
                    rec["g2"] = {"f_sup": f_, "n_frames_in_view": n_in, "per_frame": per_, "plane": {kk: (float(v) if not hasattr(v, "tolist") else v.tolist()) for kk, v in pl.items()},
                                 "tau_sup": sel.get("tau_sup"), "gated": sel.get("tau_sup") is not None}
                else:
                    rec["g2"] = {"f_sup": None, "reason": path_err or "no ground plane on the candidate", "gated": sel.get("tau_sup") is not None}
            else:
                g1["contact"] = None; g1["contact_within_tol"] = False; rec["g2"] = {"f_sup": None, "reason": "band not determinable"}
            rec["g1"] = g1
            recs.append(rec); jf.write_text(json.dumps(recs, indent=1, default=str), encoding="utf-8")
            print(f"[select/measure] {pair} seed {k}: found {rec['object_found']} IoU {rec.get('object_iou', float('nan')):.3f} m_H {rec.get('m_H', float('nan')):+.3f} CLIP {rec.get('subject_similarity', float('nan')):.3f} "
                  f"MUSIQ {rec['quality_full']:.1f} | band n {kb.get('n_band_points')} half {kb.get('half_split_rel', float('nan')):.3f} hw {kb.get('bootstrap_halfwidth_rel', float('nan')):.3f} -> {g1['band_determinable']} | "
                  f"contact {g1.get('contact') and g1['contact']['outcome']} | f_sup {rec.get('g2', {}).get('f_sup')}", flush=True)
        jf.write_text(json.dumps(recs, indent=1, default=str), encoding="utf-8")


def to_candidates(recs, sel):
    out = []
    for r in recs:
        if not r.get("object_found"):
            out.append(Candidate(r["seed"], {}, object_found=False)); continue
        dh = sel.get("delta_h"); mH = r.get("m_H")
        heading = None if dh is None else (True if mH >= float(dh) else (False if mH <= -float(dh) else None))
        g1 = r.get("g1", {}); ground_ok = bool(g1.get("band_determinable")) and bool(g1.get("contact_within_tol"))
        g2 = r.get("g2", {}); sup_ok = True if not g2.get("gated") else (g2.get("f_sup") is not None and g2["f_sup"] >= float(sel["tau_sup"]))
        c = Candidate(r["seed"], {"object_iou": r.get("object_iou"), "road_agreement": 0.0, "subject_similarity": r.get("subject_similarity"), "quality_full": r.get("quality_full")},
                      object_found=True, heading_consistent=heading)
        c.metrics["_ground_ok"] = ground_ok; c.metrics["_support_ok"] = sup_ok; c.metrics["_contact_outcome"] = (g1.get("contact") or {}).get("outcome"); c.metrics["_kbg_ci"] = g1.get("kbg_ci_rel_width"); c.metrics["_heading_tier"] = "H1 pass" if heading else ("H1 fail" if heading is False else "ambiguous -> H2 (heading unchecked)")
        out.append(c)
    return out


def select_pairs(args):
    lock_all = yaml.safe_load((ROOT / "configs" / "calibration.lock.yaml").read_text(encoding="utf-8")); sel = dict(lock_all["m2"]["selection"])
    sel.setdefault("tau_road", -1e9)                                                  # the road gate is gone (Q1): never fails
    root = Path(args.drive_root); rep = root / "reports" / "m2_selection"; rep.mkdir(parents=True, exist_ok=True)
    blind = {}
    bp = ROOT / "measurements" / "m2d_anchors" / "blind_passfail_template.csv"
    if bp.exists():
        for r in csv.DictReader(open(bp, encoding="utf-8-sig")):
            blind[(r["pair"], int(r["seed"]))] = r["verdict"].strip().lower()
    results = []
    for pair in args.pairs:
        clip, cat = pair.split("/"); d2 = root / "cache" / "m2_d" / clip / cat
        if not (d2 / "selection_v2" / "candidates.json").exists():                    # the pair never reached selection (a gate upstream, or not run yet): a row, not a crash
            print(f"[select] {pair}: no candidates measured (Module 2 did not reach selection under {root}); skipped", flush=True); continue
        recs = json.loads((d2 / "selection_v2" / "candidates.json").read_text(encoding="utf-8"))
        human = None
        qd = d2 / "qc_m2d.json"
        if qd.exists():
            qd_ = json.loads(qd.read_text(encoding="utf-8"))
            if str(qd_.get("phase2_instrument", "")).startswith("human"):              # only a human-ranked pick is 'the human's'; a v2-written qc is this rule's own output
                human = qd_.get("selected_seed")

        def run(sel_):
            cands = to_candidates(recs, sel_)
            # the two ground gates are applied here (the arithmetic module knows object/quality/heading/computable):
            for c in cands:
                if c.object_found and not c.metrics.get("_ground_ok", False):
                    c.object_found = False; c.reasons.append("ground_g1_failed")
                if c.object_found and not c.metrics.get("_support_ok", True):
                    c.object_found = False; c.reasons.append("path_support_g2_failed")
                # ambiguous heading = gate not applied (instrument-level "not computable"): pass with the flag
                if c.heading_consistent is None and c.object_found:
                    c.heading_consistent = True; c.metrics["_heading_flag"] = "heading unchecked"
            res = rule_select(cands, sel_)
            for cnd, c in zip(res["candidates"], cands):
                cnd["tier"] = c.metrics.get("_heading_tier"); cnd["ground_g1"] = c.metrics.get("_ground_ok"); cnd["path_support_g2"] = c.metrics.get("_support_ok"); cnd["extra_reasons"] = c.reasons
                cnd["contact_outcome"] = c.metrics.get("_contact_outcome"); cnd["kbg_ci_rel_width"] = c.metrics.get("_kbg_ci")
                if c.reasons:                                                        # the named gate, not the arithmetic module's proxy label
                    cnd["gates_failed"] = [g for g in cnd["gates_failed"] if g != "object_not_found"] + c.reasons
            return res
        res = run(sel)
        sens = {}
        for name, factor in (("eps_half", 0.5), ("eps_double", 2.0)):
            s2 = dict(sel); s2["eps_iou"] = float(sel["eps_iou"]) * factor; s2["eps_clip"] = float(sel["eps_clip"]) * factor
            sens[name] = run(s2)["selected"]
        pick = res["selected"]
        row = {"pair": pair, "selected": pick, "n_survivors": res["n_survivors"], "S1_seeds": res.get("S1_seeds"), "tie_broken_by_seed": res.get("tie_broken_by_seed"),
               "human_selected": human, "agrees_with_human": (pick == human) if (pick is not None and human is not None) else None,
               "blind_verdict_of_pick": blind.get((pair, pick)) if pick is not None else None,
               "human_pick_rank_note": None, "sensitivity": sens, "heading_flag": any(c.get("tier", "").startswith("ambiguous") and c["seed"] == pick for c in res["candidates"]),
               "candidates": res["candidates"], "lock": {k: sel.get(k) for k in ("eps_iou", "eps_clip", "delta_h", "tau_q", "tau_sup", "status")}}
        (d2 / "selection_v2" / "selection.json").write_text(json.dumps(row, indent=1, default=str), encoding="utf-8")
        if args.write_anchor:
            # the artefacts downstream reads (m2_e, Module 3): anchor.png + qc_m2d.json in the v1 schema, the rule recorded inside. On a
            # v1 root (the seven pairs, Module 2 closed) this is refused unless --force-v1-root: the record of the six pairs is never overwritten.
            from src.anchor.m2d import select as m2d_select
            if (d2 / "qc_m2d.json").exists() and json.loads((d2 / "qc_m2d.json").read_text(encoding="utf-8")).get("phase2_instrument", "").startswith("human") and not args.force_v1_root:
                print(f"[select] {pair}: a v1 (human-selected) anchor exists; not overwritten (pass --force-v1-root to re-select a v1 pair)", flush=True)
            else:
                gen = json.loads((d2 / "generation.json").read_text(encoding="utf-8")) if (d2 / "generation.json").exists() else {}
                prompt = (gen.get("prompt") or {}).get("prompt") if isinstance(gen.get("prompt"), dict) else gen.get("prompt")
                pr = gen.get("prompt") if isinstance(gen.get("prompt"), dict) else {}
                ctrl_cls = pr.get("heading_class"); opposite = {"toward": "away", "away": "toward", "lateral_left_to_right": "lateral_right_to_left", "lateral_right_to_left": "lateral_left_to_right"}
                dh = sel.get("delta_h"); by_seed = {r["seed"]: r for r in recs}
                def h1_class(seed):                                                    # H1: the silhouette matches the proxy -> the control's class; flipped -> its opposite; ambiguous -> None
                    mH = by_seed.get(seed, {}).get("m_H")
                    if dh is None or mH is None or ctrl_cls is None:
                        return None
                    return ctrl_cls if mH >= float(dh) else (opposite.get(ctrl_cls) if mH <= -float(dh) else None)
                cands_out = [{**c, "heading_class": h1_class(c["seed"]), "heading_instrument": "H1 silhouette margin against the control's class (None = unchecked)"} for c in res["candidates"]]
                qc = {"clip": clip, "category": cat, "polarity": gen.get("polarity"), "guidance": gen.get("guidance"), "steps": gen.get("steps"),
                      "seeds": [c["seed"] for c in gen.get("calls", [])] or [c["seed"] for c in recs], "selected_seed": pick,
                      "outcome": "selected" if pick is not None else "no_anchor", "n_survivors": res["n_survivors"], "candidates": cands_out, "prompt": prompt,
                      "heading_anchor_frame": {"class": ctrl_cls, "azimuth_deg": pr.get("heading_azimuth_deg"), "frame": pr.get("heading_frame")},
                      "dictionary_row": pr.get("dictionary_rows"), "shot_class_near_band_edge": pr.get("shot_class_near_band_edge"), "target_source": (gen.get("target") or {}).get("target_source"),
                      "heading_instrument": "H1 silhouette margin (delta_h) / H2 not yet validated -> 'heading unchecked' flag", "phase2_instrument": "selection v2: gates, eps-lexicographic proxy IoU then CLIP identity, lowest seed (rule a', reviewer 2026-10-17)",
                      "selection_v2": {k: row[k] for k in ("S1_seeds", "tie_broken_by_seed", "heading_flag", "sensitivity", "lock")},
                      "thresholds": {k: sel.get(k) for k in ("eps_iou", "eps_clip", "delta_h", "tau_q", "tau_sup")}}
                m2d_select.write_selection(d2, qc)
                print(f"[select] {pair}: wrote {'anchor.png and ' if pick is not None else ''}qc_m2d.json (selection v2)", flush=True)
        results.append(row)
        print(f"[select] {pair}: pick {pick} (survivors {res['n_survivors']}, S1 {res.get('S1_seeds')}); human {human}; blind verdict of pick: {row['blind_verdict_of_pick']}; eps/2 -> {sens['eps_half']}, eps*2 -> {sens['eps_double']}", flush=True)
    if not results:
        print("[select] no pair had measured candidates; no report written", flush=True); return 0
    (rep / "pairs").mkdir(exist_ok=True)
    for r in results:                                                                # per-pair record, so a one-pair call (the chain driver) never replaces the set's report
        (rep / "pairs" / f"{r['pair'].replace('/', '_')}.json").write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
    if args.no_report:
        return 0
    (rep / "selection.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    n_pick = sum(r["selected"] is not None for r in results); n_ok = sum(r["blind_verdict_of_pick"] == "pass" for r in results)
    gross = {}                                                                         # a 'gross' fail is a verdict whose reason is not a layout reason; the blind sheet's reasons decide
    for (pr, sd), v in blind.items():
        gross.setdefault(pr, {})[sd] = v
    chance = [len([1 for sd, v in gross[r["pair"]].items() if v == "pass"]) / len(gross[r["pair"]]) for r in results if r["pair"] in gross and gross[r["pair"]]]
    chance_hits = float(sum(chance)) if chance else None
    n_agree = sum(bool(r["agrees_with_human"]) for r in results); n_sens = sum((r["sensitivity"]["eps_half"] != r["selected"]) or (r["sensitivity"]["eps_double"] != r["selected"]) for r in results)
    short = lambda pr: (pr.split("/")[0][-9:] if pr.split("/")[0].startswith("0000") else pr.split("/")[0]) + "/" + pr.split("/")[1]   # noqa: E731
    n_human = sum(r["human_selected"] is not None for r in results)
    md = [f"# Module 2 selection v2 on {len(results)} pairs under {root} (rule a' with Q1/Q2; lock values calibrated 2026-10-18 on the seven pairs)", "",
          "| pair | survivors | S1 | pick | heading | human's pick | blind verdict of the pick | pick at eps/2 | pick at eps x2 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        md.append(f"| {r['pair']} | {r['n_survivors']} | {r['S1_seeds']} | {r['selected']} | {'unchecked' if r['heading_flag'] else 'H1'} | {r['human_selected']} | {r['blind_verdict_of_pick'] or '-'} | {r['sensitivity']['eps_half']} | {r['sensitivity']['eps_double']} |")
    g2lab = ROOT / "measurements" / "m2_selection" / "g2_selftest_labelled.json"
    g2rec = json.loads(g2lab.read_text(encoding="utf-8")) if g2lab.exists() else None
    if sel.get("tau_sup") is not None:
        g2_state = "G2 gated (tau_sup set)"
    elif g2rec is not None and not g2rec.get("selftest_passed"):
        g2_state = (f"G2 DROPPED AND DISCLOSED: its self-test failed as measured ({g2rec['aggregated']}: {len(g2rec['contact_below_0_8'])} of {g2rec['n_contact']} in-contact census clips below 0.8, "
                    f"median {g2rec['median']:.2f}; see measurements/m2_selection/g2_selftest_labelled.md). Path support is reported per candidate, never gated; the gate list is G1, H, quality.")
    else:
        g2_state = "G2 PENDING: path support reported, not gated (tau_sup awaits its self-test); the gate list is G1, H, quality — re-select once G2 exists"
    md += ["", g2_state, "", "G1 pass rate over all candidates: " + f"{sum(1 for r in results for c in r['candidates'] if c.get('ground_g1'))} of {sum(len(r['candidates']) for r in results)} (G1 guarantees admissibility to Module 3; it does not rank).",
           "Contact outcome labels of the picks: " + "; ".join(f"{short(r['pair'])}: " + str(next((c.get('contact_outcome') for c in r['candidates'] if c['seed'] == r['selected']), None)) for r in results if r['selected'] is not None),
           "", f"Anchors: {n_pick} of {len(results)} pairs got an anchor; the rest are 'no anchor' (every candidate failed a gate). Picks that change when eps is halved or doubled: {n_sens} of {n_pick}."]
    if chance:                                                                          # the seven calibration pairs carry blind verdicts; the experiment set carries none (never scored by eye)
        md += [f"Acceptable-pick rate (the thesis metric): {n_ok} of {sum(1 for r in results if r['pair'] in gross and r['selected'] is not None)} picks with a blind verdict are in the blind pass set.",
               f"Agreement with the human's selection (descriptive only): {n_agree} of {n_human}.",
               f"Chance baseline (reviewer 2026-10-19): a uniform random pick per pair is expected to land in the blind pass set {round(chance_hits, 2)} of {len(chance)} times "
               f"(sum over pairs of the pair's blind pass share: {', '.join(f'{short(r[0])} {r[1]:.2f}' for r in zip([x['pair'] for x in results if x['pair'] in gross], chance))}); the headline is read against this number."]
    else:
        md += ["No blind verdicts exist for these pairs (the experiment set is never scored by eye); the acceptable-pick rate and the chance baseline are defined on the seven calibration pairs only."]
    (rep / "selection.md").write_text("\n".join(md), encoding="utf-8"); print("\n".join(md))


def selftest(args):
    """Rule 13: G1 at candidate level must reproduce the record's admission on the six anchors of record (the human-selected seed of each
    v1 pair, whose Module 3 preconditions passed in the record). Reports, per pair, the record's k_bg CI width and contact outcome beside the
    candidate-level numbers and G1's verdict. A G1 that refuses an anchor of record has failed its known-answer test; nothing is adjusted here."""
    root = Path(args.drive_root); rows = []
    for pair in args.pairs:
        clip, cat = pair.split("/"); d2 = root / "cache" / "m2_d" / clip / cat; d3 = root / "cache" / "m3" / clip / cat
        qd = d2 / "qc_m2d.json"; human = json.loads(qd.read_text(encoding="utf-8")).get("selected_seed") if qd.exists() else None
        recs = json.loads((d2 / "selection_v2" / "candidates.json").read_text(encoding="utf-8")) if (d2 / "selection_v2" / "candidates.json").exists() else []
        cand = next((r for r in recs if r["seed"] == human), None)
        rec1 = json.loads((d3 / "sint_step1.json").read_text(encoding="utf-8")) if (d3 / "sint_step1.json").exists() else None
        cdec = json.loads((d3 / "contact_decision.json").read_text(encoding="utf-8")) if (d3 / "contact_decision.json").exists() else None
        g = (rec1 or {}).get("gates", {}).get("g_gauge_scalar", {})
        row = {"pair": pair, "human_seed": human, "record": {"kbg_ci_rel_width": g.get("ci_rel_width"), "kbg_n_pixels": g.get("n_pixels"), "contact_outcome": (cdec or {}).get("outcome"), "s_c": (cdec or {}).get("s_c"), "admitted_in_record": rec1 is not None},
               "candidate_level": None if cand is None else {"band": (cand.get("g1") or {}).get("band"), "band_determinable": (cand.get("g1") or {}).get("band_determinable"), "contact": (cand.get("g1") or {}).get("contact"), "g1_verdict": bool((cand.get("g1") or {}).get("band_determinable")) and bool((cand.get("g1") or {}).get("contact_within_tol"))}}
        row["known_answer_reproduced"] = None if (cand is None or rec1 is None) else bool(row["candidate_level"]["g1_verdict"])
        b_ = (cand.get("g1") or {}).get("band") if cand else None
        row["kbg_ci_width_candidate"] = None if not b_ or "ci95" not in b_ else float((b_["ci95"][1] - b_["ci95"][0]) / b_["k_bg"])
        row["kbg_ci_width_matches_record"] = None if (row["kbg_ci_width_candidate"] is None or g.get("ci_rel_width") is None) else bool(abs(row["kbg_ci_width_candidate"] - g["ci_rel_width"]) < 0.01)
        rows.append(row)
        print(f"[selftest] {pair} seed {human}: record k_bg CI width {g.get('ci_rel_width')}, contact {(cdec or {}).get('outcome')} | candidate-level G1 {row['candidate_level'] and row['candidate_level']['g1_verdict']} "
              f"(band {row['candidate_level'] and row['candidate_level']['band']}; contact {row['candidate_level'] and row['candidate_level']['contact']})", flush=True)
    rep = root / "reports" / "m2_selection"; rep.mkdir(parents=True, exist_ok=True)
    (rep / "g1_selftest.json").write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
    n = [r for r in rows if r["known_answer_reproduced"] is not None]
    print(f"[selftest] G1 reproduces the record's admission on {sum(r['known_answer_reproduced'] for r in n)} of {len(n)} anchors of record", flush=True)


def g2_selftest(args):
    """G2's self-test (reviewer, plan): the SOURCE object's own path on the SOURCE's own ground must read f_sup ~ 1 on every census pass.
    Per clip: Stage A's depth at t_a (the source gauge: k = 1), the ground plane fitted beside the Stage C mask, the C.5 object points per
    frame carried by camera C into camera-t_a, projected through K_a (256 grid), the lowest 5 % of the projected object tested against the
    plane inliers. tau_sup = min over clips of f_sup - 2 sigma_hat, floor 0.8 (rule in the plan); a clip whose own source reads below 0.8
    fails the self-test and G2 is dropped and disclosed. CPU, read-only; the clips come from --clips (the census passes under this root)."""
    import yaml as _y
    m3cfg = _y.safe_load((ROOT / "configs" / "m3.yaml").read_text(encoding="utf-8"))["m3"]; cc = m3cfg["contact"]
    lock_m3 = _y.safe_load((ROOT / "configs" / "calibration.lock.yaml").read_text(encoding="utf-8"))["m3"]; lock_int = lock_m3["int"]
    labels = {}
    if args.contact_labels and Path(args.contact_labels).exists():                   # John's blind labels: in_contact | not_in_contact, before any f_sup is seen
        for r in csv.DictReader(open(args.contact_labels, encoding="utf-8-sig")):
            lab = r["label"].strip().lower()
            assert lab in ("in_contact", "not_in_contact", "contact_out_of_view", ""), f"{r['clip']}: unknown contact label {lab!r}"
            labels[r["clip"]] = lab or None
    root = Path(args.drive_root); rows = []
    gate_record = {}
    for name in ("camera_c_ba.json", "camera_c.json"):                          # the gate's own held-out error per clip (bundle px), for the fixture to reproduce
        fjs = root / "reports" / "census" / name
        if fjs.exists():
            for r in json.loads(fjs.read_text(encoding="utf-8")):
                gate_record.setdefault(r["pair"].split("/")[0], (r.get("heldout_px") or r.get("heldout_reproj_px") or {}).get("median_all"))
    for clip in args.clips:
        try:
            with np.load(root / "cache" / "stage_a" / clip / "bundle.npz") as z:
                D = z["D"].astype(np.float64); K = z["K"].astype(np.float64)
            with np.load(root / "cache" / "m2_a" / clip / "stage0.npz") as z:
                t_a = int(z["t_a"])
            M, _, _ = finalize.read_mask(root / "cache" / "stage_c" / clip)
            D0 = D[t_a]; valid = np.isfinite(D0) & (D0 > 0); m_ta = M[t_a].astype(bool)
            pl, inl = ground_inliers(D0, valid, m_ta, K[t_a], cc)
            if pl is None:
                rows.append({"clip": clip, "f_sup": None, "reason": "no ground plane on the source frame"}); print(f"[g2 selftest] {clip}: no ground plane", flush=True); continue
            Hh, Ww = D0.shape; band_px = int(lock_int["contact_band_px"])
            # fixture 1 (rule 17): the composition — static tracks through G2's transform must land on their own pixel at t_a (both ways reported, the right one is small)
            fx_ok, n_ok = static_track_fixture(root, clip, t_a, int(args.frames)); fx_wrong, _ = static_track_fixture(root, clip, t_a, int(args.frames), inverse=True)
            path, err = object_path_world(root, clip, t_a, int(args.frames))
            f, n_in, per = contact_hits(path, K[t_a], pl, inl, Hh, Ww, float(cc["ransac_rel_tol"]), band_px)
            label = labels.get(clip); cam_src = load_camera_of_record(root, clip, t_a)[1]
            rec_px = gate_record.get(clip)
            rows.append({"clip": clip, "f_sup": f, "n_frames_in_view": n_in, "hit_at_t_a": per[0] if per else None, "per_frame": per, "contact_label": label,
                         "camera": cam_src, "composition_fixture_px": {"as_record": fx_ok, "wrong_way": fx_wrong, "n_frames": n_ok, "gate_heldout_median_px": rec_px, "gate_px_256": float(lock_m3["trajectory"]["camera_c_gate_patch"]) * 256.0 / int(lock_m3["lattice"]["Wp"])},
                         "plane_inlier_frac": pl["inlier_frac"], "n_inliers": pl["n_inliers"]})
            print(f"[g2 selftest] {clip} [{label or 'unlabelled'}]: f_sup {f} over {n_in} frames (t_a hit {per[0] if per else None}) | composition fixture {fx_ok} px (wrong way {fx_wrong}); plane inliers {pl['n_inliers']}", flush=True)
        except Exception as e:                                                        # noqa: BLE001
            rows.append({"clip": clip, "f_sup": None, "reason": f"{type(e).__name__}: {e}"}); print(f"[g2 selftest] {clip}: {type(e).__name__}: {e}", flush=True)
    contact = [r for r in rows if r.get("contact_label") == "in_contact" and r.get("f_sup") is not None]
    negatives = [r for r in rows if r.get("contact_label") == "not_in_contact" and r.get("f_sup") is not None]
    out_of_view = [r for r in rows if r.get("contact_label") == "contact_out_of_view"]    # on the ground, contact not in frame: reported, in neither set
    vals = np.array([r["f_sup"] for r in contact]); fx = [r for r in rows if r.get("composition_fixture_px", {}).get("as_record") is not None]
    gate_px = fx[0]["composition_fixture_px"]["gate_px_256"] if fx else None
    tau = (None if not len(vals) else max(0.8, float(vals.min() - 2 * vals.std())))
    rec = {"fixtures": {"t_a_hits": {"n_hit": sum((r.get("hit_at_t_a") or 0) for r in rows if r.get("hit_at_t_a") is not None), "n": sum(r.get("hit_at_t_a") is not None for r in rows),
                                     "reading": "at t_a the subject's lowest points lie on the ground by construction on every clip whose subject is on the ground"},
                        "composition_px": {"median_as_record": float(np.median([r["composition_fixture_px"]["as_record"] for r in fx])) if fx else None,
                                           "median_wrong_way": float(np.median([r["composition_fixture_px"]["wrong_way"] for r in fx if r["composition_fixture_px"]["wrong_way"] is not None])) if fx else None,
                                           "median_gate_record_px": float(np.median([r["composition_fixture_px"]["gate_heldout_median_px"] for r in fx if r["composition_fixture_px"]["gate_heldout_median_px"] is not None])) if any(r["composition_fixture_px"]["gate_heldout_median_px"] is not None for r in fx) else None, "camera_gate_px": gate_px, "reading": "static tracks through G2's transform must reproduce the camera gate's reprojection error (px at 256)"}},
           "contact_set": [r["clip"] for r in contact], "negative_controls": [r["clip"] for r in negatives],
           "contact_out_of_view": {r["clip"]: r.get("f_sup") for r in out_of_view}, "unlabelled": [r["clip"] for r in rows if not r.get("contact_label")],
           "rows": rows, "n_contact": int(len(vals)), "min": float(vals.min()) if len(vals) else None, "median": float(np.median(vals)) if len(vals) else None, "sigma_hat": float(vals.std()) if len(vals) else None,
           "tau_sup_rule": "over the CONTACT set (John's blind labels): min f_sup - 2 sigma_hat, floor 0.8; a contact clip below 0.8 fails the self-test -> G2 dropped and disclosed; every negative control must read below tau_sup; contact_out_of_view clips (half-body framing) are reported and enter neither set",
           "tau_sup": tau, "selftest_passed": (bool(len(vals)) and bool(vals.min() >= 0.8) and all(r["f_sup"] < (tau or 0.8) for r in negatives)),
           "contact_below_0_8": [r["clip"] for r in contact if r["f_sup"] < 0.8], "negatives_at_or_above_tau": [r["clip"] for r in negatives if tau is not None and r["f_sup"] >= tau]}
    rep_ = root / "reports" / "m2_selection"; rep_.mkdir(parents=True, exist_ok=True)
    (rep_ / "g2_selftest.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    print(f"[g2 selftest] fixtures: t_a hits {rec['fixtures']['t_a_hits']['n_hit']}/{rec['fixtures']['t_a_hits']['n']}; composition median {rec['fixtures']['composition_px']['median_as_record']} px as record vs {rec['fixtures']['composition_px']['median_wrong_way']} wrong way (gate {gate_px} px; the record's own held-out median {rec['fixtures']['composition_px']['median_gate_record_px']} px)", flush=True)
    print(f"[g2 selftest] contact set n {rec['n_contact']}, min {rec['min']}, median {rec['median']}, sigma {rec['sigma_hat']} -> tau_sup {rec['tau_sup']}; negatives {rec['negative_controls']}; passed {rec['selftest_passed']}; contact below 0.8: {rec['contact_below_0_8']}; negatives >= tau: {rec['negatives_at_or_above_tau']}; contact out of view {rec['contact_out_of_view']}; unlabelled {len(rec['unlabelled'])}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["measure", "select", "selftest", "g2selftest"])
    ap.add_argument("--drive-root", default=os.environ.get("MT_ROOT", "/content/drive/MyDrive/motion_transfer"))
    ap.add_argument("--pairs", nargs="+", default=None)
    ap.add_argument("--clips", nargs="+", default=None, help="g2selftest: clip ids (the census passes under this root)")
    ap.add_argument("--contact-labels", default="measurements/census/contact_labels.csv", help="g2selftest: John's blind labels (clip,label in_contact|not_in_contact|contact_out_of_view), written before the run")
    ap.add_argument("--checkpoint", default="Ruicheng/moge-2-vitl-normal")
    ap.add_argument("--frames", type=int, default=33, help="the window over which path support is read")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--force", action="store_true", help="measure: re-measure candidates already recorded (after a definition change)")
    ap.add_argument("--write-anchor", action="store_true", help="select: write anchor.png + qc_m2d.json (v1 schema, rule recorded) for downstream; refused on a v1 human-selected pair")
    ap.add_argument("--force-v1-root", action="store_true", help="with --write-anchor: allow re-selecting a v1 pair (reopens Module 2 for it; reviewer's call)")
    ap.add_argument("--no-report", action="store_true", help="select: write the per-pair record only, not the set-level selection.{json,md} (the chain driver's per-pair call)")
    args = ap.parse_args(argv)
    if args.mode == "g2selftest":
        assert args.clips, "--clips required"
        return g2_selftest(args)
    assert args.pairs, "--pairs required"
    return measure(args) if args.mode == "measure" else (selftest(args) if args.mode == "selftest" else select_pairs(args))


if __name__ == "__main__":
    sys.exit(main())
