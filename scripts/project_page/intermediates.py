"""Images of the pipeline's intermediates for the project page, drawn from the caches: the reference's first frame, the Stage I
subject mask on it (stage_c.npz), the lifted depth of the anchor (lift.npz), and the render target Stage III corrects toward
(render*.npz, frame 20 with the outside of the mask darkened). Every image is written once (cached by name)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

W, H = 832, 480
GOLD = np.array([232, 200, 106], dtype=np.float32)


def _save(arr: np.ndarray, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).resize((W, H), Image.BILINEAR).save(dst, quality=88, optimize=True)


def reference_frame(video: Path, dst: Path, ffmpeg: str) -> Path | None:
    if dst.exists():
        return dst
    if not video.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(video), "-vf", "select='eq(n\\,0)'", "-frames:v", "1", "-q:v", "2", str(dst)], check=True)
    return dst


def stage1_mask(stage_c: Path, frame_jpg: Path, dst: Path) -> Path | None:
    """The Stage I subject mask at the anchor frame, gold over the reference frame."""
    if dst.exists():
        return dst
    if not (stage_c.exists() and frame_jpg.exists()):
        return None
    z = np.load(stage_c)
    shape = tuple(int(v) for v in z["M_shape"])
    m = np.unpackbits(z["M_src_packed"])[: int(np.prod(shape))].reshape(shape).astype(bool)
    t = int(z["t_c"][0]) if "t_c" in z.files else 0
    mask = m[min(t, shape[0] - 1)]
    frame = np.asarray(Image.open(frame_jpg).convert("RGB").resize((W, H), Image.BILINEAR), dtype=np.float32)
    mk = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)) > 127
    out = frame.copy()
    out[mk] = 0.5 * frame[mk] + 0.5 * GOLD
    # a one-pixel edge in full gold
    edge = mk ^ np.roll(mk, 1, 0) | mk ^ np.roll(mk, 1, 1)
    out[edge] = GOLD
    _save(out, dst)
    return dst


def lifted_depth(lift: Path, dst: Path) -> Path | None:
    """Inverse depth of the lifted anchor in the source gauge, magma, invalid pixels dark."""
    if dst.exists():
        return dst
    if not lift.exists():
        return None
    import matplotlib
    cmap = matplotlib.colormaps["magma"]
    z = np.load(lift)
    d = z["depth_src_sq"].astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    if "mask_sq" in z.files:
        valid &= z["mask_sq"].astype(bool)
    inv = np.zeros_like(d)
    inv[valid] = 1.0 / d[valid]
    lo, hi = np.percentile(inv[valid], [2, 98]) if valid.any() else (0.0, 1.0)
    t = np.clip((inv - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = (cmap(t)[..., :3] * 255).astype(np.float32)
    rgb[~valid] = np.array([18, 14, 24], dtype=np.float32)
    _save(rgb, dst)
    return dst


def render_target(render: Path, dst: Path, frame: int = 20) -> Path | None:
    """The render target at `frame`, with everything outside the correction mask darkened."""
    if dst.exists():
        return dst
    if not render.exists():
        return None
    z = np.load(render)
    fr = z["frames"]
    k = min(frame, fr.shape[0] - 1)
    img = fr[k].astype(np.float32)
    m = z["mask_px"][k].astype(bool)
    img[~m] *= 0.35
    _save(img, dst)
    return dst
