"""m2_c / C3 runner — compose the hybrid control image for one clip and target category.

  python scripts/run_m2c_c3.py --clip 000000000284.0.006 --category vehicle

Reads C1's placement.json (cache/m2_c/<clip>/<category>/), the R1 proxy it names, Stage A depth + K
and the Stage C mask at t_a; runs C2 (sanitise, ungated) on the source frame; composes at 1024
under K_1024 = diag(4,4,1) K_256; writes control.npz / control.png / qc_c3.json next to the
placement record and a viewing PNG under <drive>/viz/m2c_control/. CPU only.

HARD STOP (SKILL C3 §10): human review of the composed control images before m2_d runs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.anchor.m2b2 import cache                                          # noqa: E402
from src.anchor.m2b2.objects import load_m2b2_config                       # noqa: E402
from src.anchor.m2c import compose, placement, sanitise                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True); ap.add_argument("--category", required=True)
    ap.add_argument("--occlusion-excludes-fill", action="store_true", help="R2 (reviewer 2026-10-19b): fill pixels never occlude the proxy; applied set-wide only after scripts/check_m2c_fill_occlusion.py reads 0 on every handoff")
    ap.add_argument("--config", default="configs/stage_a.yaml"); ap.add_argument("--config-m2b2", default="configs/m2b2.yaml")
    args = ap.parse_args()
    import cv2
    import yaml
    from src.preprocess.bundle import read_bundle
    from src.preprocess.config import load_config
    from src.preprocess.stage_c import finalize
    cfg_a = load_config(args.config)
    drive = str(cfg_a["paths"]["drive_root"])
    m2c_cfg = yaml.safe_load(Path("configs/m2c.yaml").read_text(encoding="utf-8"))
    cache_m2c = Path(str(m2c_cfg["paths"]["cache_m2c"]).replace("${paths.drive_root}", drive))
    out_dir = cache_m2c / args.clip / args.category
    rec = placement.read_placement(out_dir)
    print(f"[c3] placement {args.clip}/{args.category}: admitted {rec['admitted']}, z {rec['placement']['z']:.3f} s {rec['placement']['s']:.3f}, "
          f"pitch removed {rec['heading_out_of_plane_deg']:+.1f} deg, proxy {rec['proxy_cache_key']} ({rec['proxy_facing']})")
    cfg = load_m2b2_config(args.config_m2b2, drive_root=drive)
    variant = cache.route(args.category, cfg)
    prox = cache.read_proxy(Path(cfg["paths"]["cache_r1"]) / args.category / rec["proxy_cache_key"], variant, cfg["m2b2"]["adapter_version"])
    b = read_bundle(cfg_a, args.clip)
    M, present, _ = finalize.read_mask(Path(cfg_a["paths"]["stage_c_cache"]) / args.clip)
    t_a = int(rec["t_a"]) if "t_a" in rec else int(rec["source_stats"].get("t_a", 0))
    D = b["D"][t_a].astype(np.float64); K256 = b["K"][t_a].astype(np.float64); mask256 = M[t_a].astype(bool)
    st0 = rec["stage0"]
    n_plane, d_plane = np.asarray(st0["n_plane"], float), float(st0["d_plane"])
    # C2 on the source frame (ungated: thresholds None; parameters from configs/m2c.yaml)
    t0 = time.time()
    zeta_c, valid_c, rec_c2 = sanitise.sanitise(D, mask256, K256, m2c_cfg["m2c"]["c2"], thresholds=None)
    print(f"[c3] C2 sanitise {time.time() - t0:.1f}s: coarse {zeta_c.shape}, invalid frac {rec_c2['invalid_frac_coarse']:.3f}, "
          f"components {rec_c2['n_components']}, all-sky {compose.invalid_all_sky(rec_c2)}, C2 admitted {rec_c2['admitted']}")
    t0 = time.time()
    params = {**compose._c3_params(), "thresholds": cfg_a.get("thresholds", {})}        # lock-file gate values reach the gates
    fill256 = sanitise.dilate(mask256, int(m2c_cfg["m2c"]["c2"]["r"])) if args.occlusion_excludes_fill else None   # C2's own fill region (same r)
    arrays, qc = compose.compose_control(prox["vertices"], prox["faces"], rec, zeta_c, valid_c, rec_c2, K256, n_plane, d_plane, params, fill_mask_256=fill256)
    hashes = {**rec.get("upstream_hashes", {}), "c1": cache.sha256_file(out_dir / "placement.json"),
              "c2_params": m2c_cfg["m2c"]["c2"], "proxy_cache_key": rec["proxy_cache_key"]}
    f = compose.write_control(out_dir, arrays, qc, hashes)
    print(f"[c3] compose {time.time() - t0:.1f}s: emitted {qc['emitted']}; gates " +
          ", ".join(f"{k}: {v.get('outcome')}" for k, v in qc["gates"].items()))
    if not qc["emitted"]:
        print(f"[c3] refused; qc_c3.json written to {out_dir}"); return 0
    print(f"[c3] q_low {qc['q_low']:.4f} q_high {qc['q_high']:.4f}; contact step {qc['contact_step']:+.2e} vs quantum {qc['contact_step_quantum']:.2e}; "
          f"occluded {100 * qc['occluded_frac']:.1f} %; obj area {100 * qc['obj_area_frac']:.1f} %; sky {100 * qc['invalid_frac_coarse']:.1f} %; "
          f"contact in border band {qc['contact_in_border_band']}, in horizon band {qc['contact_in_horizon_band']}")
    # viewing PNG: control image with the object outline, and the source frame beside it
    viz = Path(drive) / "viz" / "m2c_control"; viz.mkdir(parents=True, exist_ok=True)
    D8 = (np.clip(arrays["D_ctrl"], 0, 1) * 255 + 0.5).astype(np.uint8)
    ctrl = np.repeat(D8[..., None], 3, 2)
    edge = arrays["M_obj"] ^ cv2.erode(arrays["M_obj"].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ctrl[edge] = [0, 255, 0]
    cap = cv2.VideoCapture(b["provenance"]["clip_path"]); cap.set(cv2.CAP_PROP_POS_FRAMES, t_a); ok, fr = cap.read(); cap.release()
    frame = cv2.resize(fr[..., ::-1], (1024, 1024), interpolation=cv2.INTER_AREA) if ok else np.zeros_like(ctrl)
    side = np.concatenate([frame, ctrl], 1)
    cv2.putText(side, f"{args.clip} {args.category}  D_ctrl (near=bright, uncalibrated)  step {qc['contact_step']:+.1e}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    png = viz / f"{args.clip}_{args.category}.png"
    cv2.imwrite(str(png), side[..., ::-1])
    print(f"[c3] wrote {f} and {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
