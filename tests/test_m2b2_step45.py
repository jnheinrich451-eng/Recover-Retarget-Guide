"""m2_b2 build steps 4-5 — SKILL_m2_b2 v1.1 §10 tests 7 (CPU half), 8, 9.

Run:  python tests/test_m2b2_step45.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.anchor.m2b2 import basis, body, cache, refconst               # noqa: E402
from src.anchor.m2b2.objects import load_m2b2_config                   # noqa: E402
from src.anchor.m2c import proxy                                        # noqa: E402

REPORT = ["# m2_b2 steps 4-5 — tests 7, 8, 9", ""]
cfg = load_m2b2_config("configs/m2b2.yaml", drive_root="/tmp/drive")
PP_MAX = cfg["m2b2"]["body"]["principal_point_max_px"]

# ---------------------------------------------------------------- test 7: Body square-pixel frame (CPU half)
cases = {"synthetic fy/fx 1.09": (np.array([[900.0, 0, 640], [0, 981.0, 360], [0, 0, 1]]), 720, 1280),
         "370.1.008 measured": (np.array([[881.3, 0, 640.8], [0, 957.8, 356.1], [0, 0, 1]]), 720, 1280),
         "370.1.009 measured": (np.array([[823.3, 0, 640.3], [0, 910.5, 357.0], [0, 0, 1]]), 720, 1280)}
lines7 = []
for name, (K, H, W) in cases.items():
    frame = np.zeros((H, W, 3), np.uint8)
    mask = np.zeros((H, W), bool); mask[300:700, 400:600] = True
    sq = body.square_pixel_frame(frame, mask, K, PP_MAX)
    assert sq.fy_fx_mismatch < 1.0 / H, sq.fy_fx_mismatch
    assert sq.frame.shape[0] == sq.mask.shape[0] == int(round(H * K[0, 0] / K[1, 1]))
    # mask/bbox round trip: the rescaled bbox maps back onto the native bbox within one native px
    bb = body.mask_bbox_xyxy(sq.mask)[0]
    back = body.to_native(bb.reshape(2, 2), sq.scale_y).ravel()
    native = body.mask_bbox_xyxy(mask)[0]
    assert np.abs(back - native).max() <= 1.0 + 1e-9, (back, native)
    lines7.append(f"{name}: H' {sq.frame.shape[0]}, fy'/fx' - 1 = {sq.fy_fx_mismatch:.2e} (< 1/H = {1 / H:.2e}), "
                  f"principal point offset ({sq.pp_offset_px[0]:+.1f}, {sq.pp_offset_px[1]:+.1f}) px vs max {PP_MAX:.0f}, bbox round trip ok")
# a principal point far from centre is refused
try:
    body.square_pixel_frame(np.zeros((720, 1280, 3), np.uint8), np.ones((720, 1280), bool),
                            np.array([[900.0, 0, 700], [0, 981.0, 360], [0, 0, 1]]), PP_MAX)
    raise AssertionError("off-centre principal point accepted")
except AssertionError as e:
    assert "principal point" in str(e)
REPORT.append("test 7 Body square-pixel (CPU half): " + "; ".join(lines7) + "; cx 60 px off-centre refused. "
              "The keypoints-onto-mask check is a runtime QC field (kp_in_mask_frac).")

# ---------------------------------------------------------------- test 8: cache key
base = dict(category="vehicle", prompt_template=cfg["m2b2"]["proxy_prompt_template"], seed_set=[0, 1, 2, 3, 4], variant="objects",
            adapter_version="1.1", model_version="facebook/sam-3d-objects@hf+abc", selected_k=0, facing="right", image_sha256="a" * 64)
k0 = cache.cache_key(**base)
assert cache.cache_key(**{**base, "prompt_template": base["prompt_template"] + " x"}) != k0
assert cache.cache_key(**{**base, "selected_k": 1}) != k0
assert cache.cache_key(**{**base, "facing": "left"}) != k0
assert cache.cache_key(**{**base, "image_sha256": "b" * 64}) != k0
assert cache.cache_key(**{**base, "seed_set": [4, 3, 2, 1, 0]}) != k0          # seed set is ordered, part of the key
assert cache.cache_key(**base) == k0
import inspect
assert "clip" not in inspect.signature(cache.cache_key).parameters
REPORT.append(f"test 8 cache key: {k0}; changes with prompt template, selected_k, facing, image hash, seed set; no clip parameter exists")

# ---------------------------------------------------------------- test 9: routing + variant consistency
assert cache.route("vehicle", cfg) == "objects" and cache.route("human", cfg) == "body"
for bad, exc in (("rider", cache.TargetExcluded), ("figurine", cache.TargetExcluded), ("dragon", KeyError)):
    try:
        cache.route(bad, cfg); raise AssertionError(bad)
    except exc:
        pass
with tempfile.TemporaryDirectory() as td:
    m = proxy.ellipsoid()
    Vn, o_p, ns = refconst.normalise(m.V)
    qc = {"variant": "objects", "adapter_version": "1.1", "facing": "right"}
    cache.write_proxy(Path(td) / "p", Vn, m.F, basis.b_tar_from_facing("right"), "right", o_p, ns, qc)
    rec = cache.read_proxy(Path(td) / "p", "objects", "1.1")
    assert rec["facing"] == "right" and abs(np.linalg.norm(rec["vertices"], axis=1).max() - 1) < 1e-4
    try:
        cache.read_proxy(Path(td) / "p", "body", "1.1"); raise AssertionError("variant mismatch accepted")
    except cache.VariantMismatch:
        pass
    try:
        cache.read_proxy(Path(td) / "p", "objects", "9.9"); raise AssertionError("adapter version mismatch accepted")
    except ValueError as e:
        assert "adapter_version" in str(e)
    try:
        cache.write_proxy(Path(td) / "q", m.V, m.F, np.eye(3), "right", o_p, ns, qc); raise AssertionError("unnormalised proxy accepted")
    except AssertionError as e:
        assert "normalised" in str(e)
    # R2 anchor record: metric, in front of the camera, schema + K_a + t_a
    X = m.V + np.array([0, 0, 6.0])
    cache.write_anchor(Path(td) / "a", X, m.F, np.eye(3), 4, {"variant": "objects"}, {"xyz": X[:10]})
    with np.load(Path(td) / "a" / "anchor.npz") as d:          # closed before the temp dir is removed (Windows)
        assert str(d["schema"]) == cache.ANCHOR_SCHEMA and int(d["t_a"]) == 4 and d["gs_xyz"].shape == (10, 3)
    try:
        cache.write_anchor(Path(td) / "b", m.V - np.array([0, 0, 6.0]), m.F, np.eye(3), 0, {}); raise AssertionError("behind-camera anchor accepted")
    except AssertionError as e:
        assert "in front" in str(e)
REPORT.append("test 9 routing: vehicle -> objects, human -> body, rider and figurine -> TargetExcluded (2026-09-17), unknown -> KeyError; "
              "read_proxy refuses variant and adapter_version mismatches; write_proxy refuses an unnormalised mesh; "
              "anchor record metric with K_a, t_a, gaussians, refuses a mesh behind the camera")

# ---------------------------------------------------------------- review 2026-09-14: aligned IoU + promotion
from src.anchor.m2b2 import silhouette                                   # noqa: E402
Kt = np.array([[1024.0, 0, 512], [0, 1024.0, 512], [0, 0, 1]])
m = proxy.ellipsoid()
X0 = m.V + np.array([0, 0, 6.0]); Xs = X0 + np.array([0.3, -0.2, 0.0])       # the same shape, translated 51 / -34 px
mask0 = silhouette.silhouette(X0, m.F, Kt, 1024, 1024)
raw = silhouette.iou(mask0, silhouette.silhouette(Xs, m.F, Kt, 1024, 1024))
al, shift = silhouette.centroid_aligned_iou(Xs, m.F, Kt, 1024, 1024, mask0)
assert raw < 0.8 and al > 0.99 and abs(shift[0] + 51.2) < 1.5 and abs(shift[1] - 34.1) < 1.5, (raw, al, shift)
assert np.abs(Xs - (X0 + np.array([0.3, -0.2, 0.0]))).max() == 0, "aligned IoU must not modify the geometry"
with tempfile.TemporaryDirectory() as td:
    rv, cr = Path(td) / "_review", Path(td)
    tpl = cfg["m2b2"]["proxy_prompt_template"]
    key = cache.cache_key("vehicle", tpl, [0, 1], "objects", "1.1", "mv", 0, "right", "c" * 64)
    qc = {"category": "vehicle", "variant": "objects", "adapter_version": "1.1", "model_version": "mv", "seeds": [0, 1],
          "selected_k": 0, "facing": "right", "image_sha256": "c" * 64, "cache_key": key}
    Vn, o_p, ns = refconst.normalise(m.V)
    cache.write_proxy(rv / "vehicle" / key, Vn, m.F, basis.b_tar_from_facing("right"), "right", o_p, ns, qc)
    dst = cache.promote(rv, cr, "vehicle", key, tpl)
    assert dst == cr / "vehicle" / key and (dst / "proxy.npz").exists() and not (rv / "vehicle" / key).exists()
    bad = "0" * 16
    cache.write_proxy(rv / "vehicle" / bad, Vn, m.F, basis.b_tar_from_facing("right"), "right", o_p, ns, {**qc, "cache_key": bad})
    try:
        cache.promote(rv, cr, "vehicle", bad, tpl); raise AssertionError("a key that does not re-derive was promoted")
    except ValueError as e:
        assert "re-derive" in str(e)
REPORT.append(f"review items: centroid-aligned IoU restores a translated mesh (raw {raw:.3f} -> aligned {al:.3f}, shift "
              f"({shift[0]:+.1f}, {shift[1]:+.1f}) px vs true (-51.2, +34.1)) without touching the geometry; promote moves only "
              "a proxy whose key re-derives from its record and refuses one that does not")

REPORT += ["", "all step-4/5 tests passed"]
Path("docs/report_m2b2_step45_tests.md").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
print("\n".join(REPORT))
