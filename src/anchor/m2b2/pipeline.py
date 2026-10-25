"""R1 (proxy, per category) and R2 (anchor, per clip) orchestration (SKILL §5, §8).

GPU paths call the reconstructor objects; everything they hand back goes through the CPU
modules (compose / refconst / basis / cache) that the tests cover.
"""
from __future__ import annotations

import platform
import time
from pathlib import Path

import numpy as np

from . import basis, cache, refconst, silhouette


def _gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:                                   # noqa: BLE001
        return "unknown"


def r1_proxy(recon, variant: str, category: str, image_rgb, mask, facing: str, image_path, selected_k: int,
             cfg: dict, out_root: Path, K_native=None) -> dict:
    """One accepted candidate image -> N seeds -> medoid by reference constants -> proxy.npz.
    `recon` is an ObjectsReconstructor (vendor_internal point map, scale-free) or a BodyReconstructor
    (its own FOV estimator when K_native is None)."""
    m = cfg["m2b2"]
    B_tar = basis.b_tar_from_facing(facing)                 # raises for toward / other
    seeds = list(range(m["seeds"]["r1"])) if variant == "objects" else [0]   # Body is deterministic
    per_seed, consts, meshes = [], [], []
    t0 = time.time()
    for sd in seeds:
        if variant == "objects":
            r = recon.reconstruct(image_rgb, mask, None, "vendor_internal", sd)
            X, F, water, nb, ne, iou = r.X_cv, r.faces, r.watertight, r.n_boundary_edges, r.n_edges, r.selfcheck_iou
            extra = {"scale_axes": r.scale_axes.tolist(), "scale": r.scale, "seconds": r.seconds}
        else:
            r = recon.reconstruct(image_rgb, mask, K_native)
            X, F = r.X_cv, r.faces
            nb, ne = silhouette.boundary_edges(F); water = nb == 0; iou = r.selfcheck_iou
            extra = {"n_people": r.n_people, "kp_in_mask_frac": r.kp_in_mask_frac, "focal_returned": r.focal_returned,
                     "seconds": r.seconds}
        Vn, o_p, ns = refconst.normalise(X)
        rc = refconst.reference_constants(Vn, F, B_tar[:, 1], m["ref_constants"]["q"], m["ref_constants"]["grid_n"])
        consts.append(rc); meshes.append((Vn, F, o_p, ns))
        per_seed.append({"seed": sd, **rc, "watertight": water, "boundary_edges": nb, "edges": ne,
                         "iou_vs_input_mask": iou, "n_vertices": int(len(X)), "n_faces": int(len(F)), **extra})
    idx, dist = refconst.medoid(refconst.medoid_features(consts))
    Vn, F, o_p, ns = meshes[idx]
    image_sha = cache.sha256_file(image_path)
    key = cache.cache_key(category, m["proxy_prompt_template"], seeds, variant, m["adapter_version"],
                          recon.model_version, selected_k, facing, image_sha)
    qc = {"category": category, "variant": variant, "cache_key": key, "seeds": seeds, "medoid_seed": seeds[idx],
          "medoid_distances": {str(seeds[i]): float(d) for i, d in enumerate(dist)}, "selected_k": selected_k,
          "facing": facing, "image_sha256": image_sha, "image_path": str(image_path), "per_seed": per_seed,
          "medoid": consts[idx], "pointmap_source": "vendor_internal" if variant == "objects" else "n/a (Body FOV estimator)",
          "scale_free": variant == "objects", "config_hash": cfg["config_hash"], "adapter_version": m["adapter_version"],
          "model_version": recon.model_version, "gpu": _gpu_name(), "host": platform.node(),
          "seconds_total": time.time() - t0, "hb_p01_minus_min": consts[idx]["ref_hb_p01_minus_min"]}
    out_dir = Path(out_root) / category / key
    cache.write_proxy(out_dir, Vn, F, B_tar, facing, o_p, ns, qc)
    # 2026-09-15 (326 figurine): the medoid by reference constants is orientation-blind, and on a blocky,
    # front/back-ambiguous object the seeds disagree on orientation (P4: 112 deg spread); the self-check
    # IoU (0.75-0.82) does not separate them either. Every seed's normalised mesh is kept so the human can
    # see which seeds kept the candidate image's profile.
    if m.get("save_seed_meshes", True):
        sd_dir = out_dir / "seeds"; sd_dir.mkdir(parents=True, exist_ok=True)
        for i, (Vn_i, F_i, _, _) in enumerate(meshes):
            np.savez_compressed(sd_dir / f"seed_{seeds[i]}.npz", vertices=np.asarray(Vn_i, np.float32),
                                faces=np.asarray(F_i, np.int32), iou=np.float32(per_seed[i]["iou_vs_input_mask"]),
                                is_medoid=np.bool_(i == idx))
    return {"out_dir": out_dir, "cache_key": key, "medoid_seed": seeds[idx], "qc": qc}


def r2_anchor(recon, variant: str, category: str, clip: str, image_rgb, mask, K_img, pointmap_cv, t_a: int,
              cfg: dict, out_root: Path, upstream_hashes: dict, K_native=None, native_frame=None, native_mask=None) -> dict:
    """Per clip. Objects: the Stage A point map (metric), N seeds, medoid by silhouette IoU against
    the anchor's own mask (ties -> lowest seed). Body: deterministic, one call on the native frame."""
    m = cache_cfg = cfg["m2b2"]
    expected = cache.route(category, cfg)
    if variant != expected:
        raise cache.VariantMismatch(f"R2 variant {variant!r} != routed {expected!r} for {category!r}")
    t0 = time.time()
    per_seed, cands = [], []
    if variant == "objects":
        H, W = mask.shape
        for sd in range(m["seeds"]["r2"]):
            r = recon.reconstruct(image_rgb, mask, pointmap_cv, "stage_a", sd, K=K_img)
            # medoid criterion = centroid-ALIGNED IoU (review 2026-09-14): the vendor's translation
            # bias varies across seeds, and raw IoU would rank by residual bias, not by shape. The
            # emitted geometry stays as the vendor placed it; the offset is a QC field.
            iou_al, shift = silhouette.centroid_aligned_iou(r.X_cv, r.faces, K_img, H, W, mask)
            per_seed.append({"seed": sd, "iou": r.selfcheck_iou, "iou_aligned": iou_al, "mask_centroid_offset_px": list(shift),
                             "watertight": r.watertight, "boundary_edges": r.n_boundary_edges,
                             "scale_axes": r.scale_axes.tolist(), "scale": r.scale, "seconds": r.seconds,
                             "t_cv": (r.X_cv.mean(0)).tolist(), "n_vertices": int(len(r.X_cv))})
            cands.append(r)
        ious = np.array([p["iou_aligned"] for p in per_seed])
        idx = int(np.flatnonzero(ious >= np.nanmax(ious) - 1e-12)[0])
        best = cands[idx]
        X, F, gs = best.X_cv, best.faces, _gaussian_arrays(best.gaussians)
        pm_src = "stage_a"
    else:
        r = recon.reconstruct(native_frame, native_mask, K_native)
        per_seed.append({"seed": 0, "iou": r.selfcheck_iou, "kp_in_mask_frac": r.kp_in_mask_frac, "n_people": r.n_people,
                         "focal_returned": r.focal_returned, "focal_echo_ok": r.extra.get("focal_echo_ok"),
                         "square_fy_fx_mismatch": r.square.fy_fx_mismatch, "square_pp_offset_px": r.square.pp_offset_px, "square_pp_offset_native_px": r.square.pp_offset_native_px, "square_pad_tblr": list(r.square.pad),
                         "seconds": r.seconds})
        idx, X, F, gs = 0, r.X_cv, r.faces, None
        pm_src = "n/a (Body: square-pixel K from Stage A)"
    qc = {"clip": clip, "category": category, "variant": variant, "seeds": [p["seed"] for p in per_seed], "medoid_seed": per_seed[idx]["seed"],
          "medoid_criterion": "iou_centroid_aligned_vs_anchor_mask" if variant == "objects" else "deterministic", "per_seed": per_seed,
          "composition_selfcheck": per_seed[idx]["iou"], "iou_aligned_medoid": per_seed[idx].get("iou_aligned"),
          "mask_centroid_offset_px": per_seed[idx].get("mask_centroid_offset_px"), "geometry": "as placed by the vendor composition (not re-centred)",
          "pointmap_source": pm_src, "t_a": int(t_a),
          "config_hash": cfg["config_hash"], "adapter_version": m["adapter_version"], "model_version": recon.model_version,
          "upstream_hashes": upstream_hashes, "gpu": _gpu_name(), "seconds_total": time.time() - t0}
    out_dir = Path(out_root) / clip
    cache.write_anchor(out_dir, X, F, K_img if variant == "objects" else K_native, t_a, qc, gs)
    return {"out_dir": out_dir, "medoid_seed": per_seed[idx]["seed"], "qc": qc}


def _gaussian_arrays(gs):
    """Best-effort extraction of the vendor's Gaussian branch; attribute names are recorded, never assumed."""
    if gs is None:
        return None
    out = {}
    for name in ("get_xyz", "_xyz", "get_scaling", "_scaling", "get_rotation", "_rotation", "get_opacity", "_opacity",
                 "get_features", "_features_dc"):
        v = getattr(gs, name, None)
        if v is None:
            continue
        try:
            arr = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            out[name.lstrip("_").replace("get_", "")] = arr
        except Exception:                                   # noqa: BLE001
            continue
    return out or None
