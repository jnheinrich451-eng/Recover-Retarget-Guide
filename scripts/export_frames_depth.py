#!/usr/bin/env python
"""Export the F frames Stage A consumed for one clip + its depth maps, for
thesis figures. Read-only on the cached bundle; no model is loaded.

    python scripts/export_frames_depth.py --clip-id 000000000035.1.003

Frames = the first F frames of the source video in reading order, exactly
as run_stage_a.load_clip takes them (F = bundle frame count, 81).
Depth  = bundle D: z-depth of camera t, model units (up to similarity),
         process res (square, anisotropic resize of the native frame),
         queried on a depth_stride grid + bilinear upsample (provenance).

Writes <drive_root>/viz/thesis/<clip_id>/:
  frames/frame_XXX.png         native-resolution RGB, lossless
  depth/depth_XXX.png          colour-mapped depth, ONE fixed scale for all
                               frames (no flicker); resized to native aspect
                               for display unless --depth-res process
  depth_z_f32.npy              raw D as float32 [F, H, W] (exact values)
  mask/mask_XXX.png            Stage C subject mask M_src, white = subject,
                               native res (nearest-neighbour, display only)
  overlay/overlay_XXX.png      frame with the mask tinted + outlined
  mask_bool.npy                raw M_src as bool [F, H, W] (process res)
  sheet_frames.{pdf,png}       contact sheet of all frames
  sheet_depth.{pdf,png}        contact sheet of all depth maps + colour bar
  sheet_mask.{pdf,png}         contact sheet of all masks
  sheet_overlay.{pdf,png}      contact sheet of all mask overlays
  frames_depth.md              manifest: indices, resolutions, scale, semantics
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess.bundle import read_bundle          # noqa: E402
from src.preprocess.config import load_config          # noqa: E402
from src.preprocess.stage_c import finalize            # noqa: E402

CMAP = "magma_r"      # perceptually uniform, monotone lightness; near = bright
PCT = (2.0, 98.0)     # fixed colour range from the clip-wide depth percentiles
MASK_RGB = (0x1b, 0xaf, 0x7a)   # same mask colour as plot_thesis_tracks.py
MASK_ALPHA = 0.40
DPI = 300


def overlay(frame, m_native):
    """Tint the mask region and draw its outline (2 px) on a copy of frame."""
    import cv2
    out = frame.astype(np.float32)
    tint = np.array(MASK_RGB, np.float32)
    out[m_native] = out[m_native] * (1 - MASK_ALPHA) + tint * MASK_ALPHA
    out = out.astype(np.uint8)
    cs, _ = cv2.findContours(m_native.astype(np.uint8), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cs, -1, MASK_RGB, 2, lineType=cv2.LINE_AA)
    return out


def read_first_frames(path: str, F: int) -> np.ndarray:
    """Same selection as scripts/run_stage_a.load_clip: frames 0..F-1."""
    import imageio.v2 as iio
    if not Path(path).exists():
        raise SystemExit(f"source clip not on this VM ({path}) - run bootstrap first")
    rd = iio.get_reader(path)
    frames = []
    for i, fr in enumerate(rd):
        if i >= F:
            break
        frames.append(fr[..., :3].copy())
    rd.close()
    assert len(frames) == F, f"video has {len(frames)} frames, bundle has {F}"
    return np.stack(frames).astype(np.uint8)


def export(D, prov, frames, out: Path, depth_res: str, cols: int, clip_id: str,
           mask=None) -> dict:
    """mask: None or (M [F,Hp,Wp] bool, object_present [F], t_c) from Stage C."""
    import cv2
    import imageio.v2 as iio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    F, Hp, Wp = D.shape
    H, W = frames.shape[1:3]
    assert frames.shape[0] == F, "frame count != depth count"
    nh = prov.get("native_hw")
    assert nh is None or tuple(nh) == (H, W), \
        f"video {H}x{W} != provenance native_hw {nh} - wrong source clip?"
    d = D.astype(np.float32)
    assert np.isfinite(d).all() and (d > 0).all(), "depth must be finite and > 0"
    lo, hi = (float(v) for v in np.percentile(d, PCT))
    cmap = plt.get_cmap(CMAP)
    norm = Normalize(lo, hi)

    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    np.save(out / "depth_z_f32.npy", d)
    col = (cmap(norm(d))[..., :3] * 255).astype(np.uint8)           # [F,Hp,Wp,3]
    for t in range(F):
        iio.imwrite(out / "frames" / f"frame_{t:03d}.png", frames[t])
        img = col[t]
        if depth_res == "native":                                 # display-only
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
        iio.imwrite(out / "depth" / f"depth_{t:03d}.png", img)

    masks = overlays = None
    if mask is not None:
        M, present, t_c = mask
        assert M.shape == D.shape and M.dtype == bool, f"mask {M.shape} vs depth {D.shape}"
        (out / "mask").mkdir(exist_ok=True)
        (out / "overlay").mkdir(exist_ok=True)
        np.save(out / "mask_bool.npy", M)
        masks, overlays = [], []
        for t in range(F):
            mn = cv2.resize(M[t].astype(np.uint8), (W, H),
                            interpolation=cv2.INTER_NEAREST).astype(bool)   # display only
            mk = (mn * 255).astype(np.uint8)
            ov = overlay(frames[t], mn)
            iio.imwrite(out / "mask" / f"mask_{t:03d}.png", mk)
            iio.imwrite(out / "overlay" / f"overlay_{t:03d}.png", ov)
            masks.append(np.repeat(mk[..., None], 3, -1))
            overlays.append(ov)

    rows = int(np.ceil(F / cols))
    tile_w = 7.4 / cols
    fig_h = rows * tile_w * H / W

    def sheet(images, name, colorbar=False):
        fig, axes = plt.subplots(rows, cols, figsize=(7.4, fig_h + (0.5 if colorbar else 0.1)))
        for k, ax in enumerate(np.atleast_1d(axes).ravel()):
            ax.axis("off")
            if k < F:
                # depth tiles drawn at the frame's aspect (extent), never resampled
                ax.imshow(images[k], extent=(0, W, H, 0), aspect="equal",
                          interpolation="nearest")
                ax.text(0.03, 0.95, str(k), transform=ax.transAxes, fontsize=5,
                        color="white", va="top",
                        bbox=dict(facecolor="black", alpha=0.5, pad=0.8, lw=0))
        fig.subplots_adjust(wspace=0.03, hspace=0.03, left=0, right=1, top=1,
                            bottom=0.06 if colorbar else 0)
        if colorbar:
            cax = fig.add_axes([0.3, 0.015, 0.4, 0.012])
            cb = fig.colorbar(ScalarMappable(norm, cmap), cax=cax, orientation="horizontal")
            cb.set_label("z-depth, model units (near = bright)", fontsize=7)
            cb.ax.tick_params(labelsize=6)
            cb.outline.set_visible(False)
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(out / f"{name}.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    sheet(frames, "sheet_frames")
    sheet(col, "sheet_depth", colorbar=True)
    if masks is not None:
        sheet(masks, "sheet_mask")
        sheet(overlays, "sheet_overlay")

    md = [f"# Frames + depth - {clip_id}", "",
          f"- frames 0..{F - 1} = first {F} frames of `{prov.get('clip_path', '?')}` in reading "
          "order (same selection as run_stage_a.load_clip)",
          f"- frame PNGs: native {W}x{H}, lossless",
          f"- depth: bundle `D`, z-depth of camera t, model units; process res {Wp}x{Hp} "
          f"(anisotropic resize, factors {prov.get('resize_factors')}); depth_stride "
          f"{prov.get('depth_stride')} + bilinear upsample (Stage A)",
          f"- depth PNGs: colour map `{CMAP}` (near = bright), fixed range "
          f"[{lo:.4f}, {hi:.4f}] = clip-wide P{PCT[0]:g}/P{PCT[1]:g}, values outside clipped; "
          + ("resized to native res with nearest-neighbour for display only"
             if depth_res == "native" else "stored at process res"),
          f"- raw depth: `depth_z_f32.npy` float32 [{F}, {Hp}, {Wp}], exact bundle values; "
          f"min {d.min():.4f}, median {np.median(d):.4f}, max {d.max():.4f}",
          f"- config hash `{prov.get('config_hash')}`"]
    if mask is not None:
        area = M.reshape(F, -1).mean(1)
        absent = [int(t) for t in np.flatnonzero(~np.asarray(present, bool))]
        md += [f"- mask: Stage C `M_src`, bool [{F}, {Hp}, {Wp}] at process res "
               f"(`mask_bool.npy`); PNGs resized to native with nearest-neighbour for "
               f"display only; overlay = tint alpha {MASK_ALPHA} + 2 px outline",
               f"- mask: canonical/prompt frame t_c = {t_c}; object_present false at "
               f"{absent if absent else 'no frame'}; mask area fraction min "
               f"{area.min():.4f}, median {np.median(area):.4f}, max {area.max():.4f}"]
    else:
        md.append("- mask: no Stage C record for this clip - not exported")
    (out / "frames_depth.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"lo": lo, "hi": hi, "md": md}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-id", default="000000000035.1.003")
    ap.add_argument("--depth-res", choices=("native", "process"), default="native")
    ap.add_argument("--cols", type=int, default=9, help="contact-sheet columns")
    ap.add_argument("--out", default=None, help="default: <drive_root>/viz/thesis/<clip>")
    args = ap.parse_args()
    cfg = load_config("configs/stage_a.yaml")
    b = read_bundle(cfg, args.clip_id)
    D, prov = b["D"], b["provenance"]
    frames = read_first_frames(prov["clip_path"], D.shape[0])
    root = Path(str(cfg["paths"]["drive_root"]))
    out = Path(args.out) if args.out else root / "viz" / "thesis" / args.clip_id
    cdir = Path(cfg["paths"]["stage_c_cache"]) / args.clip_id
    mask = finalize.read_mask(cdir) if (cdir / "qc_c.json").exists() else None
    res = export(D, prov, frames, out, args.depth_res, args.cols, args.clip_id, mask)
    print("\n".join(res["md"]))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
