"""m2_d build step 4 — SKILL_m2_d v1.1 §8 tests 2 (prompt frame) and 3 (dictionary coverage). CPU only.

Run:  python tests/test_m2d_step4.py
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.anchor import m2a                                              # noqa: E402
from src.anchor.m2d import prompt                                       # noqa: E402
from src.anchor.m2c import proxy as _proxy                              # noqa: E402
from src.anchor.m2c.synthetic import rot_y                              # noqa: E402

REPORT = ["# m2_d step 4 — anchor prompt: tests 2 (frame) and 3 (coverage)", ""]
DIC = yaml.safe_load(Path("configs/m2d.yaml").read_text(encoding="utf-8"))["m2d"]["prompt_dictionary"]
N = np.array([0.0, -1.0, 0.0])
hw = float(DIC["toward_away_half_width_deg"])


def s0(d, n=N, frame="camera_t_a"):
    return {"d_src": np.asarray(d, np.float32), "n_plane": np.asarray(n, np.float32), "frame": frame}


# ---- test 2: a heading that is LATERAL in frame 0 and AWAY in the anchor frame
# the camera yaws 90 deg between frame 0 and t_a: G[t_a] maps camera-t_a into camera-0
G = np.tile(np.eye(4), (3, 1, 1)); G[2, :3, :3] = rot_y(90.0)
d0 = np.array([1.0, 0.0, 0.0])                                   # image-right in frame 0: lateral
rg = m2a.regauge(G, 2, d_src=d0, u_scene=N, n_plane=N, d_plane=1.5)
cls0, az0 = prompt.heading_class(s0(d0), hw)                     # what the frame-0 vector WOULD select if it were mislabelled
cls_a, az_a = prompt.heading_class(s0(rg["d_src"], rg["n_plane"]), hw)
assert cls0 == "lateral_left_to_right" and cls_a == "away", (cls0, az0, cls_a, az_a)
for bad in (s0(d0, frame="camera_0"), {"d_src": d0, "n_plane": N}):   # a wrong label, and loose vectors with no label at all
    try:
        prompt.heading_class(bad, hw); raise RuntimeError("heading accepted without the camera_t_a frame")
    except prompt.WrongFrame:
        pass
REPORT.append(f"test 2 prompt frame: a heading at {az0:+.0f} deg in frame 0 ('{cls0}') is {az_a:+.0f} deg at t_a ('{cls_a}') after m2_a's "
              "re-gauge; the dictionary picks 'away'; a struct labelled camera_0 and a struct with NO frame are both refused")

# the label comes from the loader, not the caller: read_stage0 returns it and refuses any other frame on disk
with tempfile.TemporaryDirectory() as td:
    arrays = {"t_a": np.int64(2), "K_a": np.eye(3, dtype=np.float32), "d_src": rg["d_src"].astype(np.float32),
              "u_scene": N.astype(np.float32), "n_plane": rg["n_plane"].astype(np.float32), "d_plane": np.float32(1.5),
              "admitted": np.bool_(True)}
    qc = {"schema": m2a.STAGE0_SCHEMA, "frame": "camera_t_a", "heading_defined": True, "up_defined": True}
    m2a.write_stage0(Path(td), "clip", arrays, qc)
    st = m2a.read_stage0(Path(td) / "clip")
    assert st["frame"] == "camera_t_a" and prompt.heading_class(st, hw)[0] == "away"
    f = Path(td) / "clip" / "qc_m2a.json"; q = json.loads(f.read_text(encoding="utf-8")); q["frame"] = "camera_0"
    f.write_text(json.dumps(q), encoding="utf-8")
    try:
        m2a.read_stage0(Path(td) / "clip"); raise RuntimeError("read_stage0 accepted a camera_0 record")
    except AssertionError as e:
        assert "camera_t_a" in str(e), e
REPORT.append("test 2b frame as data: m2a.read_stage0 returns frame='camera_t_a' with the vectors and refuses a record whose qc says camera_0")

# ---- test 3: coverage — every heading maps to exactly one class, no default, boundaries belong to away / toward
counts = {c: 0 for c in prompt.HEADING_CLASSES}
for az in np.arange(-180.0, 180.0, 0.5):
    c, a = prompt.heading_class(s0([np.cos(np.radians(az)), 0.0, np.sin(np.radians(az))]), hw)
    assert c in counts and abs(((a - az) + 180) % 360 - 180) < 1e-4
    counts[c] += 1
assert all(v > 0 for v in counts.values()) and sum(counts.values()) == 720
at = lambda az: prompt.heading_class(s0([np.cos(np.radians(az)), 0, np.sin(np.radians(az))]), hw)[0]   # noqa: E731
assert at(45.1) == "away" and at(44.9) == "lateral_left_to_right" and at(-135.1) == "lateral_right_to_left" and at(-134.9) == "toward"
assert len({DIC["phrases"][c] for c in prompt.HEADING_CLASSES}) == 4
areas = {prompt.shot_class(a, DIC["small_area_max_frac"], DIC["medium_area_max_frac"]) for a in np.linspace(0, 1, 2001)}
assert areas == {"small_area", "medium_area", "large_area"}, areas            # every area gets a row, none falls through
# a pitched heading over a tilted plane is classified by its in-plane part (C1's rule); ILLUSTRATIVE vector
c_p, a_p = prompt.heading_class(s0([0.409, -0.195, 0.892], [0.14, -0.989, 0.036]), hw)
REPORT.append(f"test 3 coverage: 720 headings -> {counts}; boundaries belong to away / toward; four distinct heading phrases; "
              f"every area fraction in [0, 1] maps to one of three shot rows; an illustrative pitched heading is classified in-plane: '{c_p}' at {a_p:+.1f} deg")

# ---- assembly: heights DERIVED from the placement and the proxy; rows recorded; computed numbers never reach the prompt
mesh = _proxy.make_proxy("ellipsoid")                            # radii (1.3, 0.8, 0.6): extent along n = (0,-1,0) is 1.6
V_al = _proxy.aligned_vertices(mesh, np.eye(3))
pl = {"s": 1.0, "d_used": 1.2}
cam_h, obj_h = prompt.heights(pl, V_al, N)
assert abs(cam_h - 1.2) < 1e-12 and abs(obj_h - 1.6) < 1e-3, (cam_h, obj_h)
assert abs(prompt.heights({"s": 2.5, "d_used": -1.2}, V_al, N)[1] - 4.0) < 3e-3          # scales with s; the sign of d is irrelevant
lines = []
for name, d, area in (("moving away", [0.02, 0.0, 1.0], 0.03), ("moving left, slightly toward", [-0.97, 0.0, -0.26], 0.13),
                      ("moving away, small", [0.05, 0.0, 1.0], 0.015)):                # ILLUSTRATIVE headings, not measurements
    r = prompt.build_anchor_prompt("a red car", "driving", "on a mountain road at dusk", s0(d), pl, V_al, area, DIC)
    assert r["heading_frame"] == "camera_t_a" and r["dictionary_rows"][:2] == [r["heading_class"], r["shot_class"]] and not r["camera_low"]
    lines.append(f"{name}: {r['heading_class']} ({r['heading_azimuth_deg']:+.0f} deg), {r['shot_class']} -> \"{r['prompt']}\"")
ok = prompt.build_anchor_prompt("a 1967 Corvette", "driving", "on route 66", s0([0, 0, 1.0]), pl, V_al, 0.03, DIC)
assert "1967" in ok["prompt"]                                    # the user's own text is not the risk surface
bad = copy.deepcopy(DIC); bad["phrases"]["away"] = "rear view, yaw 21.3 degrees"
try:
    prompt.build_anchor_prompt("a car", "driving", "on a road", s0([0, 0, 1.0]), pl, V_al, 0.03, bad)
    raise RuntimeError("a numeric dictionary phrase was accepted")
except AssertionError as e:
    assert "numeric" in str(e)
low = prompt.build_anchor_prompt("a man", "walking", "in a forest", s0([0, 0, 1.0]), {"s": 1.0, "d_used": 0.5}, V_al, 0.03, DIC)
assert low["camera_low"] and low["dictionary_rows"][-1] == "camera_low" and low["prompt"].endswith(DIC["phrases"]["camera_low"])
REPORT.append("assembly: heights derived (camera = |d_used|, object = s x proxy extent along n; ellipsoid 1.6 recovered); "
              + "; ".join(lines) + "; 'a 1967 Corvette' passes, a dictionary phrase carrying 'yaw 21.3 degrees' is refused; "
              "camera height / object height below the ratio adds 'low-angle shot'")

REPORT += ["", "all m2_d step-4 tests passed"]
Path("docs/report_m2d_step4_tests.md").write_text("\n".join(REPORT) + "\n", encoding="utf-8")
print("\n".join(REPORT))
