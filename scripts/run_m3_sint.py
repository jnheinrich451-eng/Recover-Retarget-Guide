#!/usr/bin/env python
"""Module 3, S-int step 1 (brief v2.5 §2.5-§2.6, §3.2, §9): frame-0 lift on the frozen v1 handoffs.

Per pair: handoff gates (CPU) -> MoGe-2 depth on the square-pixel grid with the known FOV -> background scale k_bg on the
m2_e ground band (median of ratios, block bootstrap) -> background and object primitives in world = camera-t_a, source
gauge -> gates g_obj_footprint, g_obj_depth, g_gauge_scalar (record) -> cache/m3/<clip>/<category>/lift.npz and a report.
Phase 1: Wan is never loaded here. Read-only on every Module 1 / Module 2 cache. No threshold is adjusted: a failing
gate is recorded with its numbers and the pair is marked, the run continues (CLAUDE.md §3.4, §3a.18).

    pip install git+https://github.com/microsoft/MoGe.git
    python scripts/run_m3_sint.py                       # all six handed-over pairs
    python scripts/run_m3_sint.py --pairs 000000000265.2.005/vehicle
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.anchor.m2c import proxy as m2c_proxy                    # noqa: E402  (renderer only)
from src.anchor.m2e import alpha as m2e_alpha                    # noqa: E402  (the ground band's definition)
from src.generate.m3 import depth_moge, handoff as H             # noqa: E402
from src.generate.m3.geometry import project                     # noqa: E402
from src.preprocess.stage_c import finalize                      # noqa: E402

FOCAL_REL_TOL = 0.01          # the adapter's check that the model honoured the imposed FOV (interface assert, not a gate)


def resize_mask(m, W, Hh):
    import cv2
    return cv2.resize(m.astype(np.float32), (W, Hh), interpolation=cv2.INTER_AREA)


def kbg_fit(inv_zeta, D0, region, block_px, n_boot, seed):
    """k_bg = median over the band of (1/zeta_bg) / D0. Interval: bootstrap over block_px x block_px image blocks."""
    r = (inv_zeta / D0)
    ok = region & np.isfinite(r) & (r > 0)
    assert ok.sum() > 0, "empty band after the depth model's validity mask"
    k = float(np.median(r[ok]))
    vv, uu = np.nonzero(ok)
    blk = (vv // block_px) * 100000 + (uu // block_px)
    ids = np.unique(blk)
    by = {b: r[vv[blk == b], uu[blk == b]] for b in ids}
    rng = np.random.default_rng(seed)
    boots = [float(np.median(np.concatenate([by[b] for b in rng.choice(ids, len(ids))]))) for _ in range(n_boot)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    mad = float(np.median(np.abs(r[ok] - k)))
    return {"k_bg": k, "ci95": [float(lo), float(hi)], "ci_rel_width": float((hi - lo) / k), "n_pixels": int(ok.sum()),
            "n_blocks": int(len(ids)), "mad_rel": mad / k}


def run_pair(root: Path, pair: str, lock: dict, frozen, model, device: str, n_boot: int, experiment: bool = False) -> dict:
    import cv2
    clip, cat = pair.split("/")
    rp = H.read_pair(root, clip, cat, lock, frozen, experiment=experiment)
    h, P = rp["handoff"], rp["paths"]
    rec = {"pair": pair, "disposition": h["disposition"], "label": h["label"], "gates": dict(rp["gates"])}
    K1024 = np.asarray(h["K_1024"], np.float64)
    bgr = cv2.imread(str(P["anchor"])); assert bgr is not None and bgr.shape[:2] == (1024, 1024), "anchor.png missing or not 1024^2"
    qd = json.loads((root / "cache" / "m2_d" / clip / cat / "qc_m2d.json").read_text(encoding="utf-8"))
    with np.load(P["anchor_masks"]) as z:
        amask = z[f"mask_seed_{qd['selected_seed']}"].astype(bool)
        zeta_hat = z[f"zeta_hat_seed_{qd['selected_seed']}"].astype(np.float64)
    with np.load(P["control"]) as z:
        M_obj, valid_c, zeta_ctrl = z["M_obj"].astype(bool), z["valid"].astype(bool), z["zeta_ctrl"].astype(np.float64)
    M, _, _ = finalize.read_mask(root / "cache" / "stage_c" / clip)
    hole = cv2.resize(M[int(h["t_a"])].astype(np.uint8), (1024, 1024), interpolation=cv2.INTER_NEAREST).astype(bool)
    acfg = __import__("yaml").safe_load(Path("configs/m2e.yaml").read_text(encoding="utf-8"))["m2e"]["alpha"]
    band, bprov = m2e_alpha.ground_band(amask, M_obj, K1024, np.asarray(h["n_plane"], float), valid_c, hole,
                                        int(acfg["band_inner_px"]), float(acfg["band_outer_frac"]))

    # ---- depth on the square-pixel grid, FOV imposed (adapter asserts the measured conventions)
    lat = lock["lattice"]; px = int(lat["px_per_patch"]); Wn, Hn = int(lat["Wp"]) * px, int(lat["Hp"]) * px
    d = depth_moge.estimate_depth(model, bgr[..., ::-1].copy(), K1024, Wn, Hn, FOCAL_REL_TOL, device)
    Wq, Hq, Kq, D0, mk = d["W_sq"], d["H"], d["K_sq"], d["depth"], d["mask"]
    rec["grid"] = {"W_sq": Wq, "H": Hq, "fov_x_deg": d["fov_x_deg"], "fx_sq": float(Kq[0, 0]), "fy_sq": float(Kq[1, 1]),
                   "fx_over_fy_wan": float(rp["K_wan"][0, 0] / rp["K_wan"][1, 1]), "moge_valid_frac": float(mk.mean())}

    # ---- everything from the 1024^2 anchor frame is resampled ONCE to this grid (masks conservatively)
    band_q = resize_mask(band, Wq, Hq) > 0.999
    amask_q = resize_mask(amask, Wq, Hq) > 0.5
    with np.errstate(divide="ignore"):
        inv_zeta = np.where(zeta_ctrl > 0, 1.0 / zeta_ctrl, np.nan)                 # on the band zeta_ctrl IS zeta_bg (outside M_obj)
        D_alpha = 1.0 / (float(h["alpha"]) * zeta_hat + float(h["b"]))
    inv_zeta_q = cv2.resize(inv_zeta.astype(np.float32), (Wq, Hq), interpolation=cv2.INTER_AREA).astype(np.float64)
    D_alpha_q = cv2.resize(D_alpha.astype(np.float32), (Wq, Hq), interpolation=cv2.INTER_AREA).astype(np.float64)

    # ---- g_gauge_scalar (record): k_bg on the ground band
    kb = kbg_fit(inv_zeta_q, D0, band_q & mk, int(lock["int"]["kbg_block_px"]), n_boot, seed=0)
    Dsrc = kb["k_bg"] * D0
    okb = band_q & mk & np.isfinite(D_alpha_q) & (D_alpha_q > 0)
    kb["band_n_pixels_1024"] = int(bprov["n_pixels"])
    kb["alpha_bridged_over_kbg_depth_band_median"] = float(np.median(D_alpha_q[okb] / Dsrc[okb])) if okb.any() else None
    oko = amask_q & mk & np.isfinite(D_alpha_q) & (D_alpha_q > 0)
    kb["alpha_bridged_over_kbg_depth_object_median"] = float(np.median(D_alpha_q[oko] / Dsrc[oko])) if oko.any() else None
    lim = lock["int"].get("kbg_ci_rel_max")
    rec["gates"]["g_gauge_scalar"] = {**kb, "threshold": lim, "outcome": "ungated" if lim is None else ("pass" if kb["ci_rel_width"] <= lim else "fail"),
                                      "grid": f"{Hq}x{Wq} square-pixel, FOV imposed (k_bg is valid on this grid only)"}

    # ---- background primitives (never under the object mask, never where the model gives no depth)
    X_bg = depth_moge.lift_background(Dsrc, mk & ~amask_q, Kq, int(lock["lift"]["stride_px"]))

    # ---- object: G_tar as handed over (camera-t_a, source gauge, A inside). No pose, no re-centring, no A.
    with np.load(P["g_tar_mesh"]) as z:
        Vg, Fg = z["vertices"].astype(np.float64), z["faces"].astype(int)
        assert np.allclose(z["K_1024"].astype(np.float64), K1024, rtol=1e-6), "g_tar_mesh K_1024 != handoff K_1024"
    zbuf, Mg = m2c_proxy.render_perspective(Vg, Fg, K1024, 1024, 1024)
    iou = float((Mg & amask).sum() / max((Mg | amask).sum(), 1))
    th_iou = float(lock["lift"]["obj_place_iou_min"])
    rec["gates"]["g_obj_footprint"] = {"iou": iou, "handoff_R2_iou": h["R2_iou"], "threshold": th_iou, "outcome": "pass" if iou >= th_iou else "fail"}
    z_obj = float(np.median(zbuf[Mg & np.isfinite(zbuf)]))
    z_img = float(np.median(Dsrc[amask_q & mk])) if (amask_q & mk).any() else float("nan")
    rel = abs(z_obj - z_img) / z_img
    th_z = float(lock["lift"]["obj_depth_rel_tol"])
    rec["gates"]["g_obj_depth"] = {"z_gtar_visible_median": z_obj, "z_image_kbg_median": z_img, "rel_diff": rel, "signed_gtar_over_image": z_obj / z_img,
                                   "threshold": th_z, "outcome": "pass" if rel <= th_z else "fail"}

    # ---- two-grid consistency: a point lifted on the square grid lands, under K^p, on the lattice cell of its own pixel
    pb, dep = project.project(X_bg, np.eye(4)[None], rp["Kp"])
    vv, uu = np.mgrid[int(lock["lift"]["stride_px"]) // 2:Hq:int(lock["lift"]["stride_px"]), int(lock["lift"]["stride_px"]) // 2:Wq:int(lock["lift"]["stride_px"])]
    sel = (mk & ~amask_q)[vv, uu] & np.isfinite(Dsrc[vv, uu]) & (Dsrc[vv, uu] > 0)
    expect = np.stack([(uu[sel] + 0.5) * (Wn / Wq) / px, (vv[sel] + 0.5) / px], 1)
    err = float(np.abs(pb[0] - expect).max())
    assert err < 1e-6, f"square-grid lift does not land on the Wan lattice (max {err:.3e} patches): K' / K_wan inconsistent"
    Hp, Wp = int(lat["Hp"]), int(lat["Wp"])
    sky = cv2.resize((~mk).astype(np.float32), (Wp, Hp), interpolation=cv2.INTER_AREA)
    rec["lift"] = {"n_bg": int(len(X_bg)), "n_obj_vertices": int(len(Vg)), "lattice_roundtrip_max_patch": err,
                   "sky_patch_frac_gt_half": float((sky > 0.5).mean()), "c_tar_0": Vg.mean(0).tolist()}

    out = root / "cache" / "m3" / clip / cat; out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "lift.npz", X_bg=X_bg.astype(np.float32), V_obj=Vg.astype(np.float32), F_obj=Fg.astype(np.int32),
                        K_sq=Kq, K_wan=rp["K_wan"], Kp=rp["Kp"], k_bg=np.float64(kb["k_bg"]), sky_lattice=(sky > 0.5),
                        depth_src_sq=Dsrc.astype(np.float32), mask_sq=mk, anchor_mask_sq=amask_q, grid=np.array([Hq, Wq]))
    (out / "sint_step1.json").write_text(json.dumps(rec, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", default=__import__("os").environ.get("MT_ROOT", "/content/drive/MyDrive/motion_transfer"))
    ap.add_argument("--pairs", nargs="+", default=None)
    ap.add_argument("--checkpoint", default="Ruicheng/moge-2-vitl-normal")
    ap.add_argument("--manifest", default="measurements/m2_closure/m2_v1_root_hashes.json")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--experiment", action="store_true", help="the experiment path: disposition 'experiment' admitted; the frozen set is configs/experiment_handoffs.yaml, not the v1 closure manifest")
    args = ap.parse_args(argv)
    import torch
    lock = H.load_m3_lock()
    assert lock["depth_model"] == "MoGe-2-metric" and lock["depth"]["pass_fov"] is True and lock["depth"]["square_pixel_input"] is True, \
        "this script implements the locked depth interface only (MoGe-2, FOV passed, square-pixel input)"
    root = Path(args.drive_root)
    frozen = None if args.experiment else json.loads(Path(args.manifest).read_text(encoding="utf-8"))["roots"]["cache/m2_e"]["files"]
    pairs = args.pairs or sorted(f"{p.parent.parent.name}/{p.parent.name}" for p in (root / "cache" / "m2_e").glob("*/*/handoff.json"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from moge.model.v2 import MoGeModel
    model = MoGeModel.from_pretrained(args.checkpoint).to(device).eval()
    recs = []
    for pair in pairs:
        try:
            recs.append(run_pair(root, pair, lock, frozen, model, device, args.n_boot, experiment=args.experiment))
        except H.HandoffRefused as e:
            recs.append({"pair": pair, "refused": str(e)})
        except Exception as e:                                   # one pair's error must not hide the others; it is reported, not swallowed
            recs.append({"pair": pair, "error": f"{type(e).__name__}: {e}"})
        print(f"[m3 S-int] {pair}: {recs[-1].get('refused') or recs[-1].get('error') or 'done'}")
    rep = root / "reports" / "m3_sint"; rep.mkdir(parents=True, exist_ok=True)
    jf = rep / "sint_step1.json"                                                      # rule 24: one entry per pair, extended across per-pair invocations (the batch driver runs one pair at a time)
    prev = json.loads(jf.read_text(encoding="utf-8")) if jf.exists() else []
    done_ = {r["pair"] for r in recs}
    merged = [r for r in prev if r.get("pair") not in done_] + recs
    md = ["# Module 3 S-int step 1 — frame-0 lift (report; hard stop: John reviews before S-int step 2)", "",
          f"Depth model `{args.checkpoint}`, square-pixel input, FOV imposed. GPU: {torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}.", "",
          "| pair | disp. | F_pair / T_l | grid | valid frac | k_bg [CI95] | CI rel | band px / blocks | alpha-depth / k_bg-depth: band, object | footprint IoU (R2 IoU) | G_tar depth / image depth | sky patches | n_bg |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(merged, key=lambda r: r['pair']):
        if "gates" not in r or "g_gauge_scalar" not in r["gates"]:
            md.append(f"| {r['pair'][-18:]} | {r.get('refused') or r.get('error')} |"); continue
        g = r["gates"]; k = g["g_gauge_scalar"]; fp = g["g_obj_footprint"]; od = g["g_obj_depth"]; e = g["g_envelope"]
        md.append(f"| {r['pair'][-18:]} | {r['disposition']} | {e['F_pair']} / {e['T_l']} | {r['grid']['H']}x{r['grid']['W_sq']} | {r['grid']['moge_valid_frac']:.2f} | "
                  f"{k['k_bg']:.4f} [{k['ci95'][0]:.4f}, {k['ci95'][1]:.4f}] | {k['ci_rel_width']:.3f} | {k['n_pixels']} / {k['n_blocks']} | "
                  f"{k['alpha_bridged_over_kbg_depth_band_median']}, {k['alpha_bridged_over_kbg_depth_object_median']} | "
                  f"{fp['iou']:.3f} ({fp['handoff_R2_iou']:.3f}) **{fp['outcome']}** | {od['signed_gtar_over_image']:.3f} **{od['outcome']}** | "
                  f"{r['lift']['sky_patch_frac_gt_half']:.2f} | {r['lift']['n_bg']} |")
    md += ["", "Gates: footprint IoU against `lift.obj_place_iou_min`, depth agreement against `lift.obj_depth_rel_tol`; `g_gauge_scalar` is record-only "
           "(`int.kbg_ci_rel_max` is null). A failing gate is a finding with its numbers above; nothing was adjusted.",
           "", "## Decisions needed from John", "", "1. Review k_bg and its interval per pair; decide whether `int.kbg_ci_rel_max` gets a value.",
           "2. Any pair failing the footprint or depth gate: recorded refusal for Module 3, or investigate.",
           "3. Pre-registered expectation 2 (brief §14): alpha-bridged depth agrees with k_bg depth over the band within a few percent — read the column."]
    (rep / "sint_step1.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    jf.write_text(json.dumps(sorted(merged, key=lambda r: r["pair"]), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    print("\n".join(md)); print(f"-> {rep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
