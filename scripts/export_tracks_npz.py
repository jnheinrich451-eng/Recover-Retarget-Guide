#!/usr/bin/env python
"""Export one clip's pixel tracks as a single .npz for external figure
drawing: ALL sparse scene-wide tracks (Stage A, pre-selection) + the dense
subject tracks (Stage C.5). Read-only on the caches; no model is loaded.

    python scripts/export_tracks_npz.py --clip-id 000000000035.1.003

Coordinates (measured conventions, see d4rt_adapter / tracks.py):
  P_* are process-resolution pixels of the 256x256 model image, PIXEL-CENTRE
  convention: integer = pixel centre, in-frame range [0, 255] (queries are
  normalized by dim-1; P = project(K, X_local)). The 256x256 image is an
  anisotropic resize of the native frame.
  P_*_native: the same points in native-frame pixels (pixel-centre), ONLY
  when the D4RT resize mapping could be MEASURED on this VM (linear-ramp test
  on Open-d4rt's own _resize_video); otherwise the fields are absent.
Validity: Q_* = V>=tau_v & C>=tau_c & z>0 & in-frame (Stage A / C.5);
  Qvis_* additionally hides points > 8% behind the depth map (viz-only).

Writes <drive_root>/viz/thesis/<clip_id>/tracks_P_Q.npz + tracks_P_Q.md.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess.config import load_config          # noqa: E402
from scripts.plot_thesis_tracks import load, visible   # noqa: E402

# Sanity bounds on the MEASURED resize mapping (viz export, not a gate): a
# plain full-frame resize has slope n_out/n_in and a straight-line fit.
SLOPE_TOL = 0.02       # relative slope deviation -> crop/pad, refuse
FIT_RES_TOL = 0.5      # median fit residual in NATIVE px -> nonlinear, refuse
                       # (u8 ramp quantization alone is ~0.5 px max)

# Runs inside the Open-d4rt repo dir (subprocess: avoids the `src` package
# name collision with this repo). Linear-ramp test: native pixel x carries
# value x - x0 (1 level/px over 255 px). Any linear resizer (bilinear, area,
# bicubic) reproduces a linear function exactly, so an output sample j with
# value p came from native position x0 + p. A step edge does NOT work: on a
# 5x non-antialiased downscale its position is quantized to a whole output px.
MEASURE = r'''
import json, sys
import numpy as np
from infer_track_3d import _resize_video
H, W, Hp, Wp = (int(a) for a in sys.argv[1:5])
out = {}
for ax, n_in in (("u", W), ("v", H)):
    xs, js = [], []
    for x0 in np.linspace(0.1 * n_in, 0.9 * n_in - 255, 5).round().astype(int):
        ramp = np.clip(np.arange(n_in) - x0, 0, 255).astype(np.uint8)
        img = np.zeros((1, H, W, 3), np.uint8)
        if ax == "u":
            img[:] = ramp[None, None, :, None]
        else:
            img[:] = ramp[None, :, None, None]
        r = np.asarray(_resize_video(img, (Hp, Wp)))[0, ..., 0].astype(np.float64)
        prof = r.mean(0) if ax == "u" else r.mean(1)
        ok = (prof > 16) & (prof < 239)          # clear of the ramp's kinks
        js += np.flatnonzero(ok).tolist(); xs += (x0 + prof[ok]).tolist()
    xs, js = np.asarray(xs), np.asarray(js, np.float64)
    a, b = np.polyfit(xs, js, 1)
    res = np.abs((js - b) / a - xs)              # native px
    out[ax] = [float(a), float(b), float(np.median(res)), float(res.max()), int(len(xs))]
print(json.dumps(out))
'''


def measure_native_mapping(H, W, Hp, Wp, d4rt_dir):
    """-> ({'u': [a, b, res], 'v': [...]}, note) with u_proc = a * u_native + b,
    or (None, reason). Never assumed: refused unless measured and linear."""
    if not (Path(d4rt_dir) / "infer_track_3d.py").exists():
        return None, f"Open-d4rt repo not found at {d4rt_dir}"
    if min(H, W) < 320:                  # windows need 0.9n - 255 > 0.1n, i.e. n > 319
        return None, f"native {W}x{H} too small for the 255 px ramp test"
    r = subprocess.run([sys.executable, "-c", MEASURE, str(H), str(W), str(Hp), str(Wp)],
                       cwd=str(d4rt_dir), capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        tail = (r.stderr.strip().splitlines() or ["?"])[-1]
        return None, f"measurement failed: {tail}"
    m = json.loads(r.stdout.strip().splitlines()[-1])
    for ax, n_in, n_out in (("u", W, Wp), ("v", H, Hp)):
        a, b, med, mx, n = m[ax]
        if abs(a * n_in / n_out - 1) > SLOPE_TOL or med > FIT_RES_TOL:
            return None, (f"{ax}: slope {a:.5f} (plain resize {n_out / n_in:.5f}), "
                          f"median residual {med:.3f} native px - not a plain "
                          f"full-frame resize, native fields refused")
    return m, (f"measured on Open-d4rt _resize_video, ramp test "
               f"(u: {m['u'][4]} samples, median/max residual {m['u'][2]:.2f}/"
               f"{m['u'][3]:.2f} native px; v: {m['v'][4]} samples, "
               f"{m['v'][2]:.2f}/{m['v'][3]:.2f})")


def build(d: dict, subj, mapping, clip_id: str) -> dict:
    N, F = d["Q"].shape
    Hp, Wp = d["D"].shape[1:]
    c5 = d["C5"]
    lab = np.zeros(N, np.int8)                      # 0 = static background
    lab[d["B"]["y_dyn"].astype(bool)] = 2           # 2 = other dynamic-labelled
    lab[np.asarray(subj, int)] = 1                  # 1 = Stage-B subject set
    out = {
        "clip_id": np.array(clip_id),
        "process_hw": np.array([Hp, Wp], np.int32),
        "native_hw": np.array(d["prov"]["native_hw"], np.int32),
        "t_c": np.array(d["t_c"], np.int32),
        "P_sparse": d["P"].astype(np.float32),
        "Q_sparse": d["Q"].astype(bool),
        "Qvis_sparse": visible(d["D"], d["X_local"], d["P"], d["Q"]),
        "spawn_sparse": d["anchors"][:, 2].astype(np.int32),
        "label_sparse": lab,
        "P_dense": c5["P_dense"].astype(np.float32),
        "Q_dense": c5["Q_dense"].astype(bool),
        "Qvis_dense": visible(d["D"], c5["X_local_dense"], c5["P_dense"], c5["Q_dense"]),
        "spawn_dense": c5["anchors_dense"][:, 2].astype(np.int32),
    }
    assert out["P_dense"].shape[1] == F, "dense / sparse frame counts differ"
    if mapping is not None:
        ab = np.array([mapping["u"][:2], mapping["v"][:2]], np.float64)   # rows u, v
        nat = lambda P: ((P - ab[:, 1]) / ab[:, 0]).astype(np.float32)
        out["P_sparse_native"] = nat(out["P_sparse"])
        out["P_dense_native"] = nat(out["P_dense"])
        out["native_map_ab"] = ab
    return out


def manifest(o: dict, note: str, channel: str) -> list:
    N, F = o["Q_sparse"].shape
    Nd = o["Q_dense"].shape[0]
    lab = o["label_sparse"]
    Hp, Wp = (int(v) for v in o["process_hw"])
    H, W = (int(v) for v in o["native_hw"])
    subj = ("follow-cam blob fc_blob_idx (subject + halo of glued background)"
            if channel == "branch" else "X0-path winner o_src_members")
    md = [f"# tracks_P_Q.npz - {o['clip_id']}", "",
          "| key | shape | dtype | meaning |", "|---|---|---|---|"]
    doc = {
        "P_sparse": f"sparse scene-wide pixel tracks (ALL {N}, pre-selection), process px",
        "Q_sparse": "validity: V>=tau_v & C>=tau_c & z>0 & in-frame [0, 255]",
        "Qvis_sparse": "Q_sparse minus points > 8% behind the depth map (viz-only)",
        "spawn_sparse": "frame each sparse track was seeded at",
        "label_sparse": f"0 static background, 1 subject set = {subj}, 2 other dynamic",
        "P_dense": "dense subject pixel tracks under the Stage C mask, process px",
        "Q_dense": "dense validity (same four factors)",
        "Qvis_dense": "Q_dense minus occluded points (viz-only)",
        "spawn_dense": "frame each dense track was seeded at",
        "P_sparse_native": "P_sparse in native-frame pixels (measured mapping)",
        "P_dense_native": "P_dense in native-frame pixels (measured mapping)",
        "native_map_ab": "rows (u, v): u_proc = a * u_native + b (measured)",
        "process_hw": "(H, W) of the process image", "native_hw": "(H, W) of the video",
        "t_c": "Stage C canonical / prompt frame", "clip_id": "clip id"}
    for k, v in o.items():
        a = np.asarray(v)
        shp = "x".join(str(s) for s in a.shape) or "scalar"
        md.append(f"| `{k}` | `{shp}` | {a.dtype} | {doc.get(k, '')} |")
    md += ["",
           f"- F = {F} frames (index t = video frame t); N = {N} sparse "
           f"({int((lab == 0).sum())} static, {int((lab == 1).sum())} subject, "
           f"{int((lab == 2).sum())} other dynamic); N_d = {Nd} dense",
           f"- coordinates: `P[i, t] = (u, v)` on the {Wp}x{Hp} process image, pixel-centre "
           f"convention (integer = pixel centre, in-frame [0, {Wp - 1}]); the process "
           f"image is an anisotropic resize of the native {W}x{H} frame",
           f"- native mapping: {note}" + (
               f"; u_native = (u - b)/a with (a, b) u = "
               f"({o['native_map_ab'][0, 0]:.6f}, {o['native_map_ab'][0, 1]:.6f}), v = "
               f"({o['native_map_ab'][1, 0]:.6f}, {o['native_map_ab'][1, 1]:.6f})"
               if "native_map_ab" in o else
               f"; native fields OMITTED - approximate with u*{W}/{Wp}, v*{H}/{Hp} "
               f"(error <= {0.5 * W / Wp:.1f} px) and say so"),
           "- P is finite everywhere but meaningless where Q is false: always mask by Q "
           "(or Qvis for drawings)",
           f"- valid at t_c: {int(o['Q_sparse'][:, int(o['t_c'])].sum())} sparse, "
           f"{int(o['Q_dense'][:, int(o['t_c'])].sum())} dense"]
    return md


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-id", default="000000000035.1.003")
    ap.add_argument("--out", default=None, help="default: <drive_root>/viz/thesis/<clip>")
    args = ap.parse_args()
    cfg = load_config("configs/stage_a.yaml")
    c = args.clip_id
    d = load(cfg, c)
    subj = (d["B"]["fc_blob_idx"] if d["channel"] == "branch"
            else d["B"]["o_src_members"]).astype(int)
    H, W = (int(v) for v in d["prov"]["native_hw"])
    Hp, Wp = d["D"].shape[1:]
    mapping, note = measure_native_mapping(H, W, Hp, Wp, cfg["paths"]["d4rt_dir"])
    o = build(d, subj, mapping, c)
    root = Path(str(cfg["paths"]["drive_root"]))
    out = Path(args.out) if args.out else root / "viz" / "thesis" / c
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "tracks_P_Q.npz", **o)
    md = manifest(o, note, d["channel"])
    (out / "tracks_P_Q.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\n-> {out / 'tracks_P_Q.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
