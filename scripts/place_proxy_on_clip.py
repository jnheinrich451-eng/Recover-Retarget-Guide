"""Place a real R1 proxy through C1 on a real clip and render it (review, 2026-09-14): pose, solve,
rasterise, eyeball. The first time the two branches meet on real data; no composition, no FLUX.

  python scripts/place_proxy_on_clip.py --category vehicle --clip 000000000284.0.006 [--key <cache key>] [--review]

Reads: the R1 proxy (cache/m2_b2/<category>/<key>, or _review with --review), m2_a's stage0 for the
clip, Stage A depth + K and the Stage C mask at t_a. Writes <drive>/viz/m2c_place/<clip>_<category>.png
(1024x1024 overlay: green = placed proxy silhouette, red outline = source mask) and a JSON of the
placement (z, s, X, R_A, gates, refine history, centroid / contact errors).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.anchor import m2a                                                # noqa: E402
from src.anchor.m2b2 import cache, silhouette                             # noqa: E402
from src.anchor.m2b2.objects import load_m2b2_config                      # noqa: E402
from src.anchor.m2c import placement, proxy                               # noqa: E402

RENDER_PX = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True); ap.add_argument("--clip", required=True)
    ap.add_argument("--key", default=None, help="proxy cache key; default: the only key under the category")
    ap.add_argument("--proxy-gen", default=None, help="select the proxy built from this candidate, e.g. p4_human_proxy_s2")
    ap.add_argument("--review", action="store_true", help="read the proxy from <cache>/_review instead of the cache")
    ap.add_argument("--mode", default="depth_preserving", choices=list(placement.MODES) + list(placement.MODE_ALIASES))
    ap.add_argument("--grid", type=int, default=800)
    ap.add_argument("--config", default="configs/stage_a.yaml"); ap.add_argument("--config-m2b2", default="configs/m2b2.yaml")
    args = ap.parse_args()
    import cv2
    import yaml
    from src.preprocess.bundle import read_bundle
    from src.preprocess.config import load_config
    from src.preprocess.stage_c import finalize
    cfg_a = load_config(args.config)
    drive = str(cfg_a["paths"]["drive_root"])
    cfg = load_m2b2_config(args.config_m2b2, drive_root=drive)
    variant = cache.route(args.category, cfg)
    root = Path(cfg["paths"]["cache_r1"]) / ("_review" if args.review else "") / args.category
    key = cache.select_key(root, args.key, args.proxy_gen)
    rec = cache.read_proxy(root / key, variant, cfg["m2b2"]["adapter_version"])
    mesh = proxy.ProxyMesh("proxy", rec["vertices"], rec["faces"], rec["vertices"].mean(0), rec["B_tar"])
    print(f"[place] proxy {args.category}/{key}: {len(mesh.V)} vertices, facing {rec['facing']}, variant {variant}")

    m2a_out = Path(yaml.safe_load(Path("configs/m2a.yaml").read_text(encoding="utf-8"))["paths"]["out_dir"].replace("${paths.drive_root}", drive))
    st0 = m2a.read_stage0(m2a_out / args.clip)
    t_a = int(st0["t_a"])
    stage0 = {"K_a": st0["K_a"].astype(np.float64), "d_src": st0["d_src"].astype(np.float64), "u_scene": st0["u_scene"].astype(np.float64),
              "n_plane": st0["n_plane"].astype(np.float64), "d_plane": float(st0["d_plane"])}
    assert np.linalg.norm(stage0["d_src"]) > 0 and np.linalg.norm(stage0["u_scene"]) > 0, "stage0 has no heading / up (m2_a refused)"
    hoop = float(np.degrees(np.arcsin(np.clip(stage0["n_plane"] @ stage0["d_src"], -1, 1))))
    print(f"[place] stage0 t_a={t_a} admitted={bool(st0['admitted'])} d_src={np.round(stage0['d_src'], 3).tolist()} "
          f"n_plane={np.round(stage0['n_plane'], 3).tolist()} d_plane={stage0['d_plane']:.3f}; heading out of plane {hoop:+.1f} deg "
          f"(035 measured +16: the human saw the proxy tilt; 284 +0.3, 370 -4)")

    b = read_bundle(cfg_a, args.clip)
    M, present, _ = finalize.read_mask(Path(cfg_a["paths"]["stage_c_cache"]) / args.clip)
    assert bool(present[t_a]), f"subject absent at t_a={t_a}"
    D = b["D"][t_a].astype(np.float64); mask256 = M[t_a].astype(bool)
    stats = placement.source_stats(mask256, np.where(mask256 & np.isfinite(D), D, 0.0), stage0["K_a"])
    print(f"[place] source stats: centroid ({stats.u0:.1f}, {stats.v0:.1f}) area {stats.A_src:.0f} z_src {stats.z_src:.3f} "
          f"contact {'observed v=' + format(stats.v_contact, '.1f') if stats.contact_observed else 'UNOBSERVED'}")

    # contract check before solving (found 2026-09-14): the source centroid ray point at z_src must
    # lie on the +n side of its own ground plane, n.X + d > 0 (C1 §3); a negative value means the
    # plane offset's sign is wrong upstream (B.5 plane_d before the 2026-09-14 fix)
    rho = np.linalg.solve(stage0["K_a"], np.array([stats.u0, stats.v0, 1.0])); rho /= rho[2]
    above = float(stage0["n_plane"] @ (stats.z_src * rho) + stage0["d_plane"])
    # MEASURED 2026-09-14 after the B.5 fix: 284 gives -0.014 (2 % of z_src: the fitted plane passes
    # through the object's centroid ray — plane-fit noise / rough ground), 035 +0.138, 370 +0.174;
    # the sign bug gave -0.43 (65 % of z_src). Refuse only at sign-error scale; report the rest.
    rel = above / stats.z_src
    print(f"[place] plane check: n.X_centroid + d = {above:+.3f} = {100 * rel:+.1f} % of z_src (sign error scale: -65 %)")
    assert rel > -0.10, (f"source centroid is far BELOW its ground plane ({above:+.3f}, {100 * rel:+.0f} % of z_src): "
                         "stage0 plane_d sign is wrong — regenerate Stage B.5 (2026-09-14 fix) and m2_a for this clip")
    t0 = time.time()
    pl = placement.place(stage0, stats, mesh, args.mode, grid_n=args.grid)
    pl_r, hist = placement.refine(stage0, stats, mesh, pl, args.mode, iters=2)
    gates = ", ".join(f"{k}: {v.get('outcome')}" for k, v in pl_r.gates.items())
    print(f"[place] solve {time.time() - t0:.0f}s: closed form z {pl.z:.3f} s {pl.s:.3f} -> refined z {pl_r.z:.3f} s {pl_r.s:.3f}; "
          f"admitted {pl_r.admitted}; gates [{gates}]; contact residual {pl_r.contact_residual:+.4f}; pitch removed {pl_r.pitch_deg:+.1f} deg "
          f"(in-plane basis, §6.2 amended); kappa_basis {pl_r.kappa_basis:.3f}; "
          f"refine centroid err px {[round(h['centroid_err_px'], 1) for h in hist]}")

    # render at 1024 under K(1024) = diag(4,4,1) K_a on the native frame at t_a
    from src.preprocess.clip_path import resolve_clip_path
    vpath = resolve_clip_path(b["provenance"].get("clip_path"), args.clip)      # the recorded path is stale on another VM; resolved under MT_DATA
    cap = cv2.VideoCapture(str(vpath)); cap.set(cv2.CAP_PROP_POS_FRAMES, t_a); ok, fr = cap.read(); cap.release()
    assert ok, f"frame {t_a} not readable from {vpath}"
    frame = cv2.resize(fr[..., ::-1], (RENDER_PX, RENDER_PX), interpolation=cv2.INTER_AREA)
    K1024 = stage0["K_a"].copy(); K1024[:2] *= RENDER_PX / 256.0
    V = pl_r.X + pl_r.s * proxy.aligned_vertices(mesh, pl_r.R_A)
    sil = silhouette.silhouette(V, mesh.F, K1024, RENDER_PX, RENDER_PX)
    src = cv2.resize(mask256.astype(np.uint8), (RENDER_PX, RENDER_PX), interpolation=cv2.INTER_NEAREST).astype(bool)
    ry, rx = np.nonzero(sil); sy, sx = np.nonzero(src)
    err = {"centroid_err_px_1024": [float(rx.mean() - sx.mean()), float(ry.mean() - sy.mean())],
           "bottom_row_proxy_vs_source_1024": [int(ry.max()), int(sy.max())],
           "iou_vs_source_mask": silhouette.iou(src, sil), "proxy_area_over_source_area": float(sil.sum() / max(src.sum(), 1))}
    ov = frame.copy()
    ov[sil] = (0.45 * ov[sil] + [0, 140, 0]).astype(np.uint8)
    edge = src ^ cv2.erode(src.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ov[edge] = [255, 40, 40]
    cv2.putText(ov, f"{args.clip} {args.category} z={pl_r.z:.2f} s={pl_r.s:.2f} {args.mode}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    out_dir = Path(drive) / "viz" / "m2c_place"; out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{args.clip}_{args.category}.png"
    cv2.imwrite(str(png), ov[..., ::-1])
    rec_out = {"clip": args.clip, "category": args.category, "proxy_key": key, "facing": rec["facing"], "variant": variant, "t_a": t_a,
               "plane_check_centroid_above": above, "plane_check_rel_z_src": rel, "truncation_bottom_touch": not stats.contact_observed,
               "heading_out_of_plane_deg": hoop,
               "mode": args.mode, "closed_form": pl.to_json(), "refined": pl_r.to_json(), "refine_history": hist, "render_1024": err,
               "source_stats": {"u0": stats.u0, "v0": stats.v0, "A_src": stats.A_src, "z_src": stats.z_src,
                                "v_contact": stats.v_contact, "contact_observed": stats.contact_observed}}
    (out_dir / f"{args.clip}_{args.category}.json").write_text(json.dumps(rec_out, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    # C1 producer output (SKILL C3 v1.1 §6): the record C3 / m2_e / Module 3 read
    m2c_paths = yaml.safe_load(Path("configs/m2c.yaml").read_text(encoding="utf-8")).get("paths", {})
    cache_m2c = Path(str(m2c_paths.get("cache_m2c", "${paths.drive_root}/cache/m2_c")).replace("${paths.drive_root}", drive))
    hashes = {"m2a": st0["qc"].get("config_hash"), "m2b2": rec["qc"].get("config_hash"), "stage_a": b["provenance"].get("config_hash")}
    pj = placement.write_placement(cache_m2c / args.clip / args.category, pl_r, hist, args.clip, args.category, key, rec["facing"],
                                   stats, stage0, hashes, extra={"closed_form": pl.to_json(), "render_1024": err, "t_a": t_a,
                                                                  "plane_check_rel_z_src": rel})
    print(f"[place] placement record -> {pj}")
    print(f"[place] render: centroid err {np.round(err['centroid_err_px_1024'], 1).tolist()} px, bottom rows proxy/source "
          f"{err['bottom_row_proxy_vs_source_1024']}, area ratio {err['proxy_area_over_source_area']:.2f}, IoU {err['iou_vs_source_mask']:.3f}")
    print(f"[place] wrote {png} and its .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
