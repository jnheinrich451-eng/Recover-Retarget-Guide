"""SAM 3D Body adapter (SKILL §4). CPU part: the square-pixel frame. GPU part: the wrapper.

Measured (P1 body analysis, 2026-09-13): the vendor projects with ONE focal, cam_int[0,0], on
both axes and the principal point at the image centre; Stage A's K has fy/fx = 1.087 (370.1.008)
and 1.106 (370.1.009), cy 3-4 px above centre. Rescaling the frame HEIGHT by fx/fy makes the
effective fy equal fx, so the vendor's assumption holds by construction; the returned vertices
are 3D in the camera frame and need no un-scaling. Camera-frame vertices = pred_vertices +
pred_cam_t, OpenCV axes, root at the origin (E1, accepted 2026-09-14).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import silhouette


@dataclass
class SquareFrame:
    frame: np.ndarray            # uint8 H'xWx3
    mask: np.ndarray             # bool H'xW
    K: np.ndarray                # 3x3, fy == fx to 1/H (integer height), principal point as measured
    scale_y: float               # H'/H
    fy_over_fx_in: float
    fy_fx_mismatch: float        # |K'[1,1] - K'[0,0]| / K'[0,0] after the integer rescale
    pp_offset_px: tuple          # (cx - W/2, cy' - H'/2) of the frame handed to the vendor (after the re-centring pad when applied)
    pp_offset_native_px: tuple = (float("nan"), float("nan"))   # the same offset BEFORE the pad (the clip's own value, recorded)
    pad: tuple = (0, 0, 0, 0)    # (top, bottom, left, right) pixels of the re-centring pad in the rescaled frame; subtracted from 2-D outputs


def recentre_pad(frame, mask, K):
    """D7 (reviewer 2026-10-19): the vendor projects with the principal point at the image centre, so the frame is PADDED (never cropped:
    a crop can cut into a subject at the edge) asymmetrically until the principal point lands at the centre to within half a pixel, and
    K is updated by the pad (cx' = cx + pad_left, cy' = cy + pad_top). The 3-D geometry is unchanged: a world point projects to the same
    content pixel under (K, frame) and (K', padded frame), so the vendor's camera-frame vertices need no correction; its 2-D outputs
    (keypoints) are shifted back by the pad. Returns (frame', mask', K', (top, bottom, left, right))."""
    import cv2
    H, W = np.asarray(mask).shape; cx, cy = float(K[0, 2]), float(K[1, 2])
    pl = max(0, int(round(W - 2 * cx))); pr = max(0, int(round(2 * cx - W)))
    pt = max(0, int(round(H - 2 * cy))); pb = max(0, int(round(2 * cy - H)))
    fr = cv2.copyMakeBorder(frame, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    mk = cv2.copyMakeBorder(np.asarray(mask, np.uint8), pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0).astype(bool)
    Kp = np.asarray(K, np.float64).copy(); Kp[0, 2] += pl; Kp[1, 2] += pt
    Hn, Wn = mk.shape
    assert abs(Kp[0, 2] - Wn / 2.0) <= 0.5 + 1e-9 and abs(Kp[1, 2] - Hn / 2.0) <= 0.5 + 1e-9, (Kp[0, 2], Wn, Kp[1, 2], Hn)
    return fr, mk, Kp, (pt, pb, pl, pr)


def square_pixel_frame(frame_rgb, mask, K, pp_max_px: float, recentre: bool = False) -> SquareFrame:
    """Scale the height by fx/fy (rounded to an integer H'); K' = K with row 1 scaled by H'/H. With `recentre`, the rescaled frame is
    then padded so the principal point sits at the centre (recentre_pad); the pp_max_px contract is asserted on the frame handed over."""
    import cv2
    K = np.asarray(K, np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    assert fx > 0 and fy > 0 and K[0, 1] == 0 and abs(K[2, 2] - 1) < 1e-12, K
    H, W = np.asarray(mask).shape
    assert frame_rgb.shape[:2] == (H, W) and frame_rgb.dtype == np.uint8
    Hp = int(round(H * fx / fy))
    a = Hp / H
    Kp = K.copy(); Kp[1, :] *= a                       # fy' = fy a ~= fx, cy' = cy a
    mismatch = abs(Kp[1, 1] - Kp[0, 0]) / Kp[0, 0]
    assert mismatch < 1.0 / H + 1e-9, mismatch         # integer rounding bound: one part in H
    fr = cv2.resize(frame_rgb, (W, Hp), interpolation=cv2.INTER_AREA if a < 1 else cv2.INTER_LINEAR)
    mk = cv2.resize(np.asarray(mask, np.uint8), (W, Hp), interpolation=cv2.INTER_NEAREST).astype(bool)
    pp_native = (float(Kp[0, 2] - W / 2.0), float(Kp[1, 2] - Hp / 2.0)); pad = (0, 0, 0, 0)
    if recentre:
        fr, mk, Kp, pad = recentre_pad(fr, mk, Kp); Hp, W = mk.shape
    pp = (float(Kp[0, 2] - W / 2.0), float(Kp[1, 2] - Hp / 2.0))
    assert max(abs(pp[0]), abs(pp[1])) <= pp_max_px, \
        f"principal point {pp} px from centre after the rescale exceeds {pp_max_px} px (SKILL §4)"
    return SquareFrame(fr, mk, Kp, a, fy / fx, mismatch, pp, pp_native, pad)


def to_native(uv_square, scale_y):
    """Points in the rescaled frame -> native frame (x unchanged, y / a)."""
    p = np.asarray(uv_square, np.float64).copy()
    p[..., 1] /= scale_y
    return p


def mask_bbox_xyxy(mask):
    ys, xs = np.nonzero(np.asarray(mask, bool))
    assert len(xs), "empty mask"
    return np.array([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], np.float32)


@dataclass
class BodyResult:
    X_cv: np.ndarray             # [V,3] f64 camera frame, metres
    faces: np.ndarray
    pred_cam_t: np.ndarray
    global_rot: np.ndarray       # axis-angle (QC only, never B_tar)
    focal_returned: float
    n_people: int
    person_index: int
    seconds: float
    selfcheck_iou: float = float("nan")
    kp_in_mask_frac: float = float("nan")
    square: SquareFrame | None = None
    extra: dict = field(default_factory=dict)


class BodyReconstructor:
    """Base env (the Body deps + SAM2 from scripts/setup_sam3d.sh)."""

    def __init__(self, cfg: dict):
        import os
        import sys
        from pathlib import Path
        self.cfg = cfg
        bc = cfg["m2b2"]["body"]
        repo = Path(os.environ.get("SAM3D_BODY_DIR", "/content/sam-3d-body"))
        if not repo.exists():
            raise FileNotFoundError(f"SAM 3D Body repo not found at {repo} — run scripts/setup_sam3d.sh")
        sys.path.insert(0, str(repo))
        # The probe's loader (P0-P4): the vendor's notebook/utils.py setup_sam_3d_body wires the
        # ViTDet person detector and the MoGe2 FOV estimator; the package's load_sam_3d_body_hf
        # returns the bare estimator (measured 2026-09-14: the package has no setup_sam_3d_body).
        import importlib.util
        spec = importlib.util.spec_from_file_location("sam3d_body_notebook_utils", repo / "notebook" / "utils.py")
        utils = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(repo / "notebook"))
        spec.loader.exec_module(utils)
        t0 = time.time()
        self.est = utils.setup_sam_3d_body(hf_repo_id=bc["hf_repo"])
        self.load_seconds = time.time() - t0
        self.faces = np.asarray(self.est.faces, int)
        self.model_version = bc["hf_repo"]

    def reconstruct(self, frame_rgb, mask, K_native=None, recentre: bool = True) -> BodyResult:
        """K_native given (R2, real frame): square-pixel rescale + cam_int. K_native None (R1,
        generated image): the vendor's own FOV estimator; the self-check then uses the vendor's
        camera (returned focal, image centre)."""
        import torch
        bc = self.cfg["m2b2"]["body"]
        if K_native is not None:
            sq = square_pixel_frame(frame_rgb, mask, K_native, bc["principal_point_max_px"], recentre=recentre)   # D7: every clip, one pipeline (False: the fixture's A run only)
            cam = torch.from_numpy(sq.K.astype(np.float32))[None]      # batched (1,3,3): the vendor indexes [:, ...]
        else:
            H0, W0 = np.asarray(mask).shape
            sq = SquareFrame(frame_rgb, np.asarray(mask, bool), None, 1.0, float("nan"), float("nan"), (float("nan"), float("nan")))
            cam = None
        bbox = mask_bbox_xyxy(sq.mask)
        t0 = time.time()
        outs = self.est.process_one_image(sq.frame, bboxes=bbox, masks=sq.mask.astype(np.uint8), cam_int=cam)
        dt = time.time() - t0
        if not outs:
            raise RuntimeError("SAM 3D Body returned no person for the mask bbox")
        Hp, W = sq.mask.shape
        if sq.K is None:                                                # vendor camera: f = returned focal, centre
            fv = float(outs[0]["focal_length"])
            sq.K = np.array([[fv, 0, W / 2.0], [0, fv, Hp / 2.0], [0, 0, 1.0]])
        best, bi, best_mk = -1.0, 0, None
        for j, o in enumerate(outs):                                    # pick by overlap with the mask
            Vc = np.asarray(o["pred_vertices"], np.float64) + np.asarray(o["pred_cam_t"], np.float64)
            mk = silhouette.silhouette(Vc, self.faces, sq.K, Hp, W)
            io = silhouette.iou(sq.mask, mk)
            if io > best:
                best, bi, best_mk = io, j, mk
        o = outs[bi]
        Vc = np.asarray(o["pred_vertices"], np.float64) + np.asarray(o["pred_cam_t"], np.float64)
        kp = np.asarray(o["pred_keypoints_2d"], np.float64)            # rescaled (padded) frame coords
        inside = (kp[:, 0] >= 0) & (kp[:, 0] < W) & (kp[:, 1] >= 0) & (kp[:, 1] < Hp)
        ii = np.clip(np.round(kp).astype(int), [0, 0], [W - 1, Hp - 1])
        import cv2
        tol = cv2.dilate(sq.mask.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)
        frac = float((tol[ii[:, 1], ii[:, 0]] & inside).sum() / max(inside.sum(), 1))
        res = BodyResult(Vc, self.faces, np.asarray(o["pred_cam_t"], np.float64), np.asarray(o["global_rot"], np.float64),
                         float(o["focal_length"]), len(outs), bi, dt, best, frac, sq)
        res.extra["kp_native"] = to_native(kp - np.array([sq.pad[2], sq.pad[0]], np.float64), sq.scale_y).tolist()   # pad removed, then the height rescale
        res.extra["recentre"] = {"pad_tblr": list(sq.pad), "pp_offset_native_px": list(sq.pp_offset_native_px), "pp_offset_handed_px": list(sq.pp_offset_px)}
        res.extra["kp3d_cam"] = (np.asarray(o["pred_keypoints_3d"], np.float64) + np.asarray(o["pred_cam_t"], np.float64)).tolist()
        res.extra["focal_echo_ok"] = abs(res.focal_returned - sq.K[0, 0]) < 1e-2 if K_native is not None else None
        return res
