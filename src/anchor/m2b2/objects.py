"""SAM 3D Objects wrapper (SKILL §2, §3). GPU; imports the vendor lazily and only inside the
Objects micromamba env (python 3.11, pytorch3d, kaolin, flash_attn — scripts/setup_sam3d.sh).
Everything CPU-side (composition, point maps, self-check) lives in the sibling modules so the
tests never touch this file's imports.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from . import compose, pointmap, silhouette

_M2B2_CFG = Path("configs/m2b2.yaml")


def load_m2b2_config(path=_M2B2_CFG, drive_root: str | None = None) -> dict:
    """The adapter's own yaml (no thresholds; the lock file defines none for m2_b2 yet).
    `${paths.drive_root}` is filled from Stage A's config when drive_root is given."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    paths = dict(raw.get("paths", {}))
    if drive_root is not None:
        paths = {k: str(v).replace("${paths.drive_root}", str(drive_root)) for k, v in paths.items()}
    paths["objects_repo"] = os.environ.get("SAM3D_OBJECTS_DIR", paths.get("objects_repo", "/content/sam-3d-objects"))
    return {"m2b2": raw["m2b2"], "paths": paths, "config_hash": m2b2_config_hash(raw["m2b2"])}


def m2b2_config_hash(section: dict) -> str:
    blob = json.dumps({"m2b2": section}, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class ObjectsResult:
    """One vendor call, composed into the OpenCV camera frame of the input image."""
    X_cv: np.ndarray                 # [V,3] f64, metric iff the point map was metric
    faces: np.ndarray                # [F,3] int
    V_local: np.ndarray              # [V,3] raw FlexiCubes vertices (unit cube)
    q_wxyz: np.ndarray               # (4,) as returned
    t_p3d: np.ndarray                # (3,) as returned (PyTorch3D frame)
    scale: float                     # scalar after the isotropy gate
    scale_axes: np.ndarray           # (3,) as returned, for the QC record
    R_cam_local: np.ndarray          # QC only — never B_tar (SKILL §3)
    gaussians: object | None         # the vendor's gs branch (kept, SKILL §8)
    seed: int
    seconds: float
    pointmap_source: str
    n_boundary_edges: int = -1
    n_edges: int = -1
    selfcheck_iou: float = float("nan")
    extra: dict = field(default_factory=dict)

    @property
    def watertight(self) -> bool:
        return self.n_boundary_edges == 0


class ObjectsReconstructor:
    """Holds the loaded vendor pipeline; one instance per process."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        oc = cfg["m2b2"]["objects"]
        repo = Path(cfg["paths"]["objects_repo"])
        if not repo.exists():
            raise FileNotFoundError(f"SAM 3D Objects repo not found at {repo} — run scripts/setup_sam3d.sh")
        # The vendor's notebook/inference.py needs CONDA_PREFIX + LIDRA_SKIP_INIT before importing
        # sam3d_objects; we run the env python directly (measured 2026-09-12).
        if not Path(sys.prefix, "lib").joinpath("python3.11").exists() or "sam3d" not in sys.prefix:
            raise RuntimeError(f"run the Objects adapter with the env python, not {sys.executable} "
                               "(e.g. /content/micromamba/envs/sam3d-objects/bin/python)")
        os.environ.setdefault("CONDA_PREFIX", sys.prefix)
        os.environ.setdefault("LIDRA_SKIP_INIT", "true")
        sys.path.insert(0, str(repo))
        import importlib.util
        spec = importlib.util.spec_from_file_location("sam3d_objects_inference", repo / "notebook" / "inference.py")
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(repo / "notebook"))
        spec.loader.exec_module(mod)
        pipe_cfg = repo / "checkpoints" / oc["ckpt_tag"] / "pipeline.yaml"
        if not pipe_cfg.exists():
            raise FileNotFoundError(f"weights missing: {pipe_cfg} (gated: {oc['hf_repo']})")
        t0 = time.time()
        self.model = mod.Inference(str(pipe_cfg), compile=False)
        self.load_seconds = time.time() - t0
        import torch
        from pytorch3d.transforms import Transform3d
        from sam3d_objects.pipeline.inference_pipeline_pointmap import camera_to_pytorch3d_camera
        self._torch = torch
        self._conv = Transform3d().rotate(camera_to_pytorch3d_camera(device="cuda").rotation).to("cuda")
        self.model_version = self._vendor_version(repo, oc)

    @staticmethod
    def _vendor_version(repo: Path, oc: dict) -> str:
        head = repo / ".git" / "HEAD"
        commit = "unknown"
        try:
            ref = head.read_text().strip()
            commit = (repo / ".git" / ref.split(" ", 1)[1]).read_text().strip()[:12] if ref.startswith("ref:") else ref[:12]
        except OSError:
            pass
        return f"{oc['hf_repo']}@{oc['ckpt_tag']}+{commit}"

    def _to_p3d(self, pm_cv):
        t = self._torch.from_numpy(np.asarray(pm_cv, np.float32)).to("cuda")
        return self._conv.transform_points(t.reshape(-1, 3)).reshape(t.shape)

    def reconstruct(self, image_rgb, mask, pointmap_cv, source: str, seed: int, K=None) -> ObjectsResult:
        """image_rgb uint8 HxWx3, mask bool HxW, pointmap_cv HxWx3 OpenCV (or None only with
        source == 'vendor_internal'), K for the runtime self-check (None -> fitted from the vendor
        map on vendor_internal paths)."""
        oc = self.cfg["m2b2"]["objects"]
        mask = np.asarray(mask, bool)                       # bool: a 0/255 uint8 overflows in the vendor (P0)
        assert image_rgb.dtype == np.uint8 and image_rgb.shape[:2] == mask.shape, (image_rgb.shape, mask.shape)
        assert mask.any(), "empty mask"
        pm = pointmap.require_pointmap(pointmap_cv, source)
        rgba = self.model.merge_mask_to_rgba(image_rgb, mask)
        t0 = time.time()
        od = self.model._pipeline.run(rgba, None, int(seed), stage1_only=oc["stage1_only"],
                                      with_mesh_postprocess=oc["with_mesh_postprocess"],
                                      with_texture_baking=oc["with_texture_baking"],
                                      with_layout_postprocess=oc["with_layout_postprocess"],
                                      use_vertex_color=oc["use_vertex_color"], stage1_inference_steps=None,
                                      pointmap=None if pm is None else self._to_p3d(pm))
        dt = time.time() - t0
        m0 = od["mesh"][0]
        V = m0.vertices.detach().float().cpu().numpy().astype(np.float64)
        F = m0.faces.detach().cpu().numpy().astype(np.int64)
        q, t, sc = (od[k].detach().cpu().numpy().reshape(-1).astype(np.float64) for k in ("rotation", "translation", "scale"))
        X, Rcl, _, s = compose.compose_objects(V, q, t, sc, oc["scale_rel_tol"])
        nb, ne = silhouette.boundary_edges(F)
        res = ObjectsResult(X, F, V, q, t, s, sc, Rcl, od.get("gs"), int(seed), dt, source, nb, ne)
        # runtime self-check (SKILL §10 test 1): the placed mesh against the input mask under K
        H, W = mask.shape
        if K is None and source == "vendor_internal" and "pointmap" in od:
            vm = od["pointmap"]; vm = vm.detach().cpu().numpy() if hasattr(vm, "detach") else np.asarray(vm)
            vm = (vm.reshape(-1, 3) @ compose.P3D_TO_CV).reshape(vm.shape)
            if vm.shape[:2] != (H, W):
                import cv2
                vm = cv2.resize(vm.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
            K = pointmap.fit_pinhole(vm, mask)
            res.extra["K_fitted"] = K.tolist()
        if K is not None and (X[:, 2] > 0).mean() > 0.5:
            res.selfcheck_iou = silhouette.iou(mask, silhouette.silhouette(X, F, K, H, W))
        return res
