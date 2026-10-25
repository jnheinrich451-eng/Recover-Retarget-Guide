#!/usr/bin/env python
"""Render target per pair (reviewer 2026-10-10; v2.7 pending): the anchor's own background, lifted densely from the pair's source-gauge depth on
the square-pixel grid, rendered through the bridged camera E'_t (geometry cache) with a z-buffer into every video frame, with the mask
M_t = rendered & not object (G_tar's primitives on the translation-only path, dilated) & not sky, eroded by one latent pixel. Uncovered pixels
are filled with the anchor frame (the VAE needs a full image; the mask keeps them out of the correction). Latent-grid masks: all four frames of
a chunk covered, min-pooled 8x8. Phase 1, CPU, read-only on the caches; writes cache/m3/<clip>/<cat>/render{tag}.npz + render{tag}.mp4 and a
review contact sheet under reports/m3_s4/.

    python scripts/render_m3_target.py --pairs 000000000265.2.005/vehicle --bridge rec
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_m3_s2_readout import WAN_H, WAN_W                               # noqa: E402
from src.generate.m3 import handoff as H, render_target as RT                    # noqa: E402

LATENT_PX = 8


def run_pair(root: Path, pair: str, bridge: str, erode_px: int, obj_dilate_px: int, splat: int, with_object: bool = False, bg_exclude_px: int = 6, obj_z_tol: float = 0.03,
             sky_in_mask: bool = False, exclude_mask_path=None, demo=None) -> dict:
    """with_object (reviewer 2026-10-11, object channel): the object's own appearance is rendered too. Its points are the anchor's object pixels
    lifted on the same grid (I_tar at the frame-0 projection, every point visible at frame 0), moved rigidly along the retargeted path
    X_t = R_t (X - c_0) + c_t with R_t = I (admitted: yaw change 0.9 deg < turn_min_deg 15 on this pair), drawn into the same z-buffer as the
    background. Its mask is where the object wins the z-buffer at t, eroded by erode_px; it is kept apart from the background mask so the sampler
    can apply its own gamma_obj. Output: render{tag}_obj.npz (frames with the object; mask_px / mask_latent = background as before; mask_obj_px /
    mask_obj_latent = the object)."""
    import cv2
    clip, cat = pair.split("/"); d = root / "cache" / "m3" / clip / cat
    tag = "" if bridge == "rec" else f"_beta_{bridge}"
    with np.load(d / "lift.npz") as z:
        Dsrc, mk, amask, Kq, K_wan, sky_l = z["depth_src_sq"].astype(np.float64), z["mask_sq"].astype(bool), z["anchor_mask_sq"].astype(bool), z["K_sq"].astype(np.float64), z["K_wan"].astype(np.float64), z["sky_lattice"].astype(bool)
        Hq, Wq = (int(v) for v in z["grid"])
    with np.load(d / f"geometry{tag}.npz") as z:
        E_full, c_tar, V_obj, T_l, F_pair = z["E_full"].astype(np.float64), z["c_tar"].astype(np.float64), z["V_obj_prims"].astype(np.float64), int(z["T_l"]), int(z["F_pair"])
        beta_tag = str(z["bridge_run"]) if "bridge_run" in z.files else "v2.5"
    assert E_full.shape[0] == F_pair and np.allclose(E_full[0], np.eye(4), atol=1e-9)
    anchor = cv2.imread(str(demo["anchor"] if demo else root / "cache" / "m2_d" / clip / cat / "anchor.png")); assert anchor is not None, "anchor unreadable"
    anchor = anchor[..., ::-1]
    corridor = None
    if demo:
        # the demo object's mask (the probe's SAM2 segmentation), dilated, on the square grid; its corridor = the reference object's 2-D displacement
        dm_ = cv2.imread(str(demo["object_mask"]), cv2.IMREAD_GRAYSCALE); assert dm_ is not None, demo["object_mask"]
        k_dm = np.ones((2 * int(demo["dilate_px"]) + 1, 2 * int(demo["dilate_px"]) + 1), np.uint8)
        demo_mask_sq = cv2.dilate(cv2.resize(dm_, (Wq, Hq), interpolation=cv2.INTER_NEAREST), k_dm) > 127
        demo_mask_wan0 = cv2.dilate(cv2.resize(dm_, (WAN_W, WAN_H), interpolation=cv2.INTER_NEAREST), k_dm) > 127
        with np.load(demo["corridor"]) as zc:
            corridor = zc["disp_px"].astype(np.float64)                                   # [F_pair, 2] Wan px, image-space displacement of the reference object
        assert len(corridor) >= F_pair, (len(corridor), F_pair)
        amask = amask | demo_mask_sq                                                       # the demo object's pixels never enter the lift (their depth is the proxy's)
        print(f"[render demo] anchor {Path(demo['anchor']).name}; object mask {demo_mask_sq.mean():.3f} of the grid (dilated {demo['dilate_px']} px); corridor max "
              f"{np.linalg.norm(corridor[:F_pair], axis=1).max() / 16:.2f} patch", flush=True)
    anchor_sq = cv2.resize(anchor, (Wq, Hq), interpolation=cv2.INTER_AREA)         # the square-pixel grid the depth lives on
    anchor_wan = cv2.resize(anchor, (WAN_W, WAN_H), interpolation=cv2.INTER_AREA)  # the Wan frame (fill for uncovered pixels)
    sky_wan = cv2.resize(sky_l.astype(np.uint8), (WAN_W, WAN_H), interpolation=cv2.INTER_NEAREST).astype(bool)
    nodepth_wan = cv2.resize((~mk).astype(np.uint8), (WAN_W, WAN_H), interpolation=cv2.INTER_NEAREST).astype(bool)   # the anchor's pixels with no depth: the sky, carried at infinity
    dyn_wan = None
    if exclude_mask_path:
        dm = cv2.imread(str(exclude_mask_path), cv2.IMREAD_GRAYSCALE); assert dm is not None, f"{exclude_mask_path} unreadable"
        dyn_wan = cv2.resize(dm, (WAN_W, WAN_H), interpolation=cv2.INTER_NEAREST) > 127      # dynamic texture (reviewer 2026-10-13 check 4): Wan's, out of the mask
        print(f"[render] dynamic-texture exclusion: {dyn_wan.mean():.3f} of the frame (static pixels still render; the correction leaves them to Wan)", flush=True)
    am_x = cv2.dilate(amask.astype(np.uint8), np.ones((2 * bg_exclude_px + 1, 2 * bg_exclude_px + 1), np.uint8)).astype(bool) if bg_exclude_px > 0 else amask
    X, C = RT.lift_dense(Dsrc, mk & ~am_x, Kq, anchor_sq)                          # background only, a bg_exclude_px ring around the object left out (its depth is the object's shadow)
    Xo = Co = vis0 = None
    if with_object:
        from src.anchor.m2c import proxy as m2c_proxy
        with np.load(root / "cache" / "m3" / clip / cat / "lift.npz") as z:
            Vm, Fm = z["V_obj"].astype(np.float64), z["F_obj"].astype(int)       # G_tar as handed over: camera-t_a frame, source gauge, placed
        zbuf_o, _ = m2c_proxy.render_perspective(Vm, Fm, Kq, Hq, Wq)              # the mesh's own z-buffer on the square grid at frame 0
        pv = (Kq @ Vm.T).T; uo, vo = pv[:, 0] / pv[:, 2], pv[:, 1] / pv[:, 2]
        ui, vi = np.clip(np.floor(uo).astype(int), 0, Wq - 1), np.clip(np.floor(vo).astype(int), 0, Hq - 1)
        inside = (uo >= 0) & (uo < Wq) & (vo >= 0) & (vo < Hq) & (Vm[:, 2] > 1e-6)
        Xo, Co = Vm[inside], anchor_sq[vi[inside], ui[inside]]                      # every vertex coloured from the anchor at its frame-0 projection
        zb = zbuf_o[vi[inside], ui[inside]]
        vis0 = np.isfinite(zb) & (Xo[:, 2] <= zb * (1.0 + obj_z_tol) + 1e-6)        # visible at frame 0: in front of the mesh's own surface there
        print(f"[render] object: {len(Xo)} mesh vertices coloured from the anchor, {int(vis0.sum())} visible at frame 0", flush=True)
    # the object path at full rate: linear interpolation of the latent-frame centroid path (translation-only, R_t = I)
    tl = np.arange(T_l) * 4
    shift_full = np.stack([np.interp(np.arange(F_pair), tl, c_tar[:, i] - c_tar[0, i]) for i in range(3)], 1)
    k_er = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
    frames = np.zeros((F_pair, WAN_H, WAN_W, 3), np.uint8); masks = np.zeros((F_pair, WAN_H, WAN_W), bool); cov = np.zeros(F_pair)
    dyn_t = None
    if dyn_wan is not None:
        # carry the dynamic-region label with the geometry: render a label image (255 = dynamic) through the same cameras
        lab_sq = cv2.resize(dyn_wan.astype(np.uint8) * 255, (Wq, Hq), interpolation=cv2.INTER_NEAREST)
        Xl_, Cl_ = RT.lift_dense(Dsrc, mk & ~am_x, Kq, np.repeat(lab_sq[..., None], 3, 2))
        dyn_t = np.zeros((F_pair, WAN_H, WAN_W), bool)
        for t_ in range(F_pair):
            li, lc, _ = RT.render_frame(Xl_, Cl_, E_full[t_], K_wan, WAN_H, WAN_W, splat=splat)
            dyn_t[t_] = lc & (li[..., 0] > 127)
    masks_obj = np.zeros((F_pair, WAN_H, WAN_W), bool) if with_object else None
    covered_all = np.zeros((F_pair, WAN_H, WAN_W), bool)
    fp_all = np.zeros((F_pair, WAN_H, WAN_W), bool)                                      # the dilated object footprint per frame: hard zero in the soft mask there
    for t in range(F_pair):
        img, covered, depth = RT.render_frame(X, C, E_full[t], K_wan, WAN_H, WAN_W, splat=splat)
        if corridor is not None:
            dx, dy = corridor[t]
            M_ = np.float32([[1, 0, dx], [0, 1, dy]])
            fp = cv2.warpAffine(demo_mask_wan0.astype(np.uint8), M_, (WAN_W, WAN_H), flags=cv2.INTER_NEAREST, borderValue=0).astype(bool)   # hard zero: the mask moved along the corridor
        else:
            fp = RT.object_footprint(V_obj, shift_full[t], E_full[t], K_wan, WAN_H, WAN_W, depth, dilate_px=obj_dilate_px)
        m = covered & ~fp & ~sky_wan
        if dyn_wan is not None:
            # the exclusion travels with the camera: the frame-0 dynamic pixels are lifted and re-projected like the rest (nearest covered-pixel label)
            m = m & ~dyn_t[t]
        m = cv2.erode(m.astype(np.uint8), k_er).astype(bool) if erode_px > 0 else m
        fp_all[t] = fp
        sky_t = None
        if t > 0:
            # the fill (reviewer 2026-10-11 §3): never the anchor at the same pixel. Sky (no depth) = the anchor's sky at infinity, moved by the
            # camera's rotation only; every other uncovered pixel = push-pull of the BACKGROUND render of this frame (the object is composited after)
            sky_img, sky_t = RT.sky_at_infinity(anchor_wan, nodepth_wan, E_full[t], K_wan)
            img = RT.pushpull_fill(img, covered)
            use_sky = ~covered & sky_t
            img = np.where(use_sky[..., None], sky_img, img)
            if sky_in_mask:
                # reviewer 2026-10-12 §5: the sky is the one region with an exact target (infinite depth: only the rotation moves it), so it joins the
                # correction mask; the object footprint and the erosion still apply
                m_sky = cv2.erode((use_sky & ~fp).astype(np.uint8), k_er).astype(bool) if erode_px > 0 else (use_sky & ~fp)
                m = m | m_sky
        if with_object:
            # the object's points moved along the path (R_t = I), then everything through one z-buffer; the object owns the pixels it wins
            img_o, cov_o, depth_o = RT.render_frame(Xo + shift_full[t][None], Co, E_full[t], K_wan, WAN_H, WAN_W, splat=splat)
            won = cov_o & (depth_o <= depth * (1.0 + obj_z_tol) + 1e-6)             # the object wins ties within obj_z_tol of the background depth
            _, cov_v, depth_v = RT.render_frame(Xo[vis0] + shift_full[t][None], Co[vis0], E_full[t], K_wan, WAN_H, WAN_W, splat=splat)
            won_mask = won & cov_v & (depth_v <= depth_o * (1.0 + obj_z_tol) + 1e-6)   # the mask: pixels whose winning surface was visible at frame 0
            img = np.where(won[..., None], img_o, img); covered = covered | cov_o
            mo = cv2.erode(won_mask.astype(np.uint8), k_er).astype(bool) if erode_px > 0 else won_mask
            masks_obj[t] = mo
            m = m & ~won                                                                # a pixel belongs to one mask only
        frames[t] = img if t > 0 else np.where(covered[..., None], img, anchor_wan)   # frame 0 IS the anchor; later frames are filled above
        if t == 0:
            covered0 = covered.copy()
        covered_all[t] = covered
        masks[t] = m; cov[t] = float(covered.mean())
    # latent-grid masks: chunk 0 = frame 0 alone; chunk j >= 1 = frames 4j-3 .. 4j (Wan's causal 4:1 temporal packing)
    def to_latent(mp):
        lat_ = np.zeros((T_l, WAN_H // LATENT_PX, WAN_W // LATENT_PX), bool)
        lat_[0] = RT.latent_mask(mp[0:1], LATENT_PX)
        for j in range(1, T_l):
            lat_[j] = RT.latent_mask(mp[4 * j - 3: 4 * j + 1], LATENT_PX)
        return lat_
    def to_latent_soft(mp):
        """The fraction of each latent cell covered: mean over the chunk's frames of the mean over its 8x8 block (reviewer 2026-10-12 §1)."""
        def frac(block):                                                             # block [n, H, W] bool -> [H/8, W/8] float
            m_ = np.asarray(block, np.float32).mean(0)
            return m_.reshape(WAN_H // LATENT_PX, LATENT_PX, WAN_W // LATENT_PX, LATENT_PX).mean((1, 3))
        lat_ = np.zeros((T_l, WAN_H // LATENT_PX, WAN_W // LATENT_PX), np.float32)
        lat_[0] = frac(mp[0:1])
        for j in range(1, T_l):
            lat_[j] = frac(mp[4 * j - 3: 4 * j + 1])
        return lat_
    lat = to_latent(masks); lat_soft = to_latent_soft(masks)
    # refined soft mask (reviewer 2026-10-12 §2): soft weights at coverage boundaries, HARD ZERO in any latent cell whose block touches the
    # dilated object footprint in any frame of its chunk (the fill ring round the object must never be corrected toward)
    def footprint_cells(fpx):
        def anyblock(block):
            m_ = np.asarray(block, bool).any(0)
            return m_.reshape(WAN_H // LATENT_PX, LATENT_PX, WAN_W // LATENT_PX, LATENT_PX).any((1, 3))
        c_ = np.zeros((T_l, WAN_H // LATENT_PX, WAN_W // LATENT_PX), bool)
        c_[0] = anyblock(fpx[0:1])
        for j in range(1, T_l):
            c_[j] = anyblock(fpx[4 * j - 3: 4 * j + 1])
        return c_
    fp_cells = footprint_cells(fp_all)
    lat_soft = np.where(fp_cells, 0.0, lat_soft).astype(np.float32)
    bg_zeroed_by_obj = 0.0
    if with_object:                                                                    # one mask per latent cell as per pixel: the object wins where it was rendered (396.3.002, 2026-10-19)
        obj_soft_ = to_latent_soft(masks_obj); both_ = (obj_soft_ > 0) & (lat_soft > 0)
        bg_zeroed_by_obj = float(both_.mean()); lat_soft = np.where(both_, 0.0, lat_soft).astype(np.float32); lat = lat & ~both_
    osuf = ("_obj" if with_object else "") + ("_sky" if sky_in_mask else "") + ("_nodyn" if exclude_mask_path else "") + ("_demo" if demo else "")
    extra = {"mask_obj_px": masks_obj, "mask_obj_latent": to_latent(masks_obj), "mask_obj_latent_soft": to_latent_soft(masks_obj), "object_points": len(Xo), "object_points_visible_at_0": int(vis0.sum()),
             "object_source": "G_tar mesh vertices coloured from the anchor at their frame-0 projection; mask = surfaces visible at frame 0",
             "object_rotation": "identity (admitted: yaw change below turn_min_deg)"} if with_object else {}
    np.savez_compressed(d / f"render{tag}{osuf}.npz", frames=frames, mask_px=masks, mask_latent=lat, mask_latent_soft=lat_soft, coverage=cov, beta_tag=beta_tag, erode_px=erode_px, obj_dilate_px=obj_dilate_px,
                        fill="push-pull of the frame's own background render; sky (no depth) = anchor sky at infinity under the camera rotation; frame 0 = anchor", chunking="frame0 | 4j-3..4j",
                        covered_px0=covered0, covered_px=covered_all, bg_exclude_px=bg_exclude_px, obj_z_tol=obj_z_tol, sky_in_mask=sky_in_mask,
                        soft_mask_rule="fraction covered; hard zero in cells touching the dilated object footprint (reviewer 2026-10-12); background zeroed where the object latent mask is nonzero (one mask per cell)",
                        bg_cells_zeroed_by_object_frac=np.float64(bg_zeroed_by_obj), **extra)
    if with_object:
        masks = masks | masks_obj                                                         # the review video darkens what neither mask covers
    vw = cv2.VideoWriter(str(d / f"render{tag}{osuf}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 16, (WAN_W, WAN_H))
    for t in range(F_pair):
        f_ = frames[t].copy(); f_[~masks[t]] = (0.5 * f_[~masks[t]]).astype(np.uint8)        # the review video darkens what the mask excludes
        vw.write(f_[..., ::-1])
    vw.release()
    rep = root / "reports" / "m3_s4"; rep.mkdir(parents=True, exist_ok=True)
    idx = [0, F_pair // 4, F_pair // 2, 3 * F_pair // 4, F_pair - 1]
    tiles = []
    for t in idx:
        f_ = frames[t].copy(); f_[~masks[t]] = (0.5 * f_[~masks[t]]).astype(np.uint8)
        f_ = cv2.resize(f_, (416, 240)); cv2.putText(f_, f"f{t} cov {cov[t]:.2f} mask {masks[t].mean():.2f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2); tiles.append(f_)
    cv2.imwrite(str(rep / f"render_{clip}_{cat}{tag}{osuf}.png"), np.hstack(tiles)[..., ::-1])
    rec = {"pair": pair, "bridge": beta_tag + osuf, "sky_in_mask": bool(sky_in_mask), "demo": ({k: str(v) for k, v in demo.items()} if demo else None), "exclude_mask": (str(exclude_mask_path) if exclude_mask_path else None), "soft_mask_zeroed_cells_frac": float(fp_cells.mean()), "F_pair": F_pair, "T_l": T_l, "n_points": int(len(X)), "coverage_px_mean": float(cov.mean()), "coverage_px_last": float(cov[-1]),
           "fill": "push-pull of the frame's background render; sky at infinity (rotation-only warp of the anchor sky)",
           **({"object_points": int(len(Xo)), "mask_obj_px_mean": float(masks_obj.mean()), "mask_obj_px_last": float(masks_obj[-1].mean()),
               "mask_obj_latent_mean": float(extra["mask_obj_latent"].mean()), "mask_obj_latent_last": float(extra["mask_obj_latent"][-1].mean())} if with_object else {}),
           "mask_px_mean": float(masks.mean()), "mask_px_last": float(masks[-1].mean()), "mask_latent_mean": float(lat.mean()), "mask_latent_last": float(lat[-1].mean()),
           "mask_latent_per_frame": [float(v) for v in lat.mean((1, 2))], "mask_latent_soft_mean": float(lat_soft.mean()), "mask_latent_soft_last": float(lat_soft[-1].mean()),
           "erode_px": erode_px, "obj_dilate_px": obj_dilate_px, "splat": splat}
    print(f"[render] {pair} ({beta_tag}): {len(X)} points; pixel coverage mean {cov.mean():.2f} last {cov[-1]:.2f}; latent mask mean {lat.mean():.2f} last {lat[-1].mean():.2f}")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", default=__import__("os").environ.get("MT_ROOT", "/content/drive/MyDrive/motion_transfer"))
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--bridge", default="rec", choices=["rec", "mid", "far"])
    ap.add_argument("--erode-px", type=int, default=LATENT_PX, help="one latent pixel (reviewer)")
    ap.add_argument("--obj-dilate-px", type=int, default=8)
    ap.add_argument("--splat", type=int, default=1, help="render each point into a (2s+1)^2 block; 1 closes pinholes of the stride-4 cloud")
    ap.add_argument("--object", action="store_true", help="also render the object's own appearance along the path (R_t = I); writes render{tag}_obj.npz")
    ap.add_argument("--bg-exclude-px", type=int, default=6, help="ring around the anchor's object mask left out of the background lift (its depth is the object's shadow)")
    ap.add_argument("--obj-z-tol", type=float, default=0.03, help="the object wins the z-buffer within this relative depth of the background")
    ap.add_argument("--exclude-mask", default=None, help="PNG (white = dynamic texture: water, foliage, smoke, clouds; Wan frame size) taken out of the correction mask; writes render{tag}..._nodyn.npz")
    ap.add_argument("--demo-anchor", default=None, help="UNREGISTERED demonstration: this image is the anchor (e.g. the bear candidate), with --demo-object-mask and --demo-corridor")
    ap.add_argument("--demo-object-mask", default=None, help="the demo object's mask PNG (white = object) at the anchor's size")
    ap.add_argument("--demo-corridor", default=None, help="demo_corridor.npz from scripts/make_m3_demo_corridor.py (the reference object's 2-D displacement)")
    ap.add_argument("--demo-dilate-px", type=int, default=12)
    ap.add_argument("--sky-in-mask", action="store_true", help="add the sky at infinity (rotation-only warp of the anchor's no-depth pixels) to the correction mask; writes render{tag}[_obj]_sky.npz")
    args = ap.parse_args(argv)
    H.load_m3_lock(); root = Path(args.drive_root)
    demo = None
    if args.demo_anchor:
        assert args.demo_object_mask and args.demo_corridor and not args.object, "a demo render needs --demo-object-mask and --demo-corridor and no --object"
        demo = {"anchor": args.demo_anchor, "object_mask": args.demo_object_mask, "corridor": args.demo_corridor, "dilate_px": args.demo_dilate_px}
    rows = [run_pair(root, p, args.bridge, args.erode_px, args.obj_dilate_px, args.splat, with_object=args.object, bg_exclude_px=args.bg_exclude_px, obj_z_tol=args.obj_z_tol, sky_in_mask=args.sky_in_mask, exclude_mask_path=args.exclude_mask, demo=demo) for p in args.pairs]
    rep = root / "reports" / "m3_s4"; rep.mkdir(parents=True, exist_ok=True)
    jf = rep / "render_target.json"
    prev = json.loads(jf.read_text(encoding="utf-8")) if jf.exists() else []
    key = lambda r: (r["pair"], r["bridge"])                                                 # noqa: E731
    rows = sorted([r for r in prev if key(r) not in {key(x) for x in rows}] + rows, key=key)
    jf.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    md = ["# Module 3 — render target per pair (reviewer 2026-10-10): the anchor's background moved by the bridged camera", "",
          "| pair | bridge | points | pixel coverage mean / last | mask (px) mean / last | latent mask mean / last | erode px | object dilate px |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['pair'][-22:]} | {r['bridge']} | {r['n_points']} | {r['coverage_px_mean']:.2f} / {r['coverage_px_last']:.2f} | {r['mask_px_mean']:.2f} / {r['mask_px_last']:.2f} | {r['mask_latent_mean']:.2f} / {r['mask_latent_last']:.2f} | {r['erode_px']} | {r['obj_dilate_px']} |")
    md += ["", "Review at full resolution: `render_<clip>_<cat>.png` (darkened = outside the mask) and `render.mp4` in the pair's cache folder. The mask is where the correction applies; "
           "everything else is left to the model. Seams at the mask edge and render artefacts are the known risks (reviewer); the pixel grader stays the arbiter."]
    (rep / "render_target.md").write_text("\n".join(md) + "\n", encoding="utf-8"); print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
