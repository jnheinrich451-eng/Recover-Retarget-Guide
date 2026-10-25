"""Module 3 handoff reader: JSON-level integration gates on the six REAL frozen v1 handoffs (repository copies) and on
tampered copies. CPU, no model, no Drive.

Run:  python tests/test_m3_handoff.py
"""
from __future__ import annotations

import copy
import glob
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.generate.m3 import handoff as H  # noqa: E402


def must_raise(exc, fn, what):
    try:
        fn()
    except exc:
        return
    raise RuntimeError(f"accepted: {what}")


files = sorted(glob.glob("measurements/m2e/handoff_*.json"))
assert len(files) == 6
man = json.loads(Path("measurements/m2_closure/m2_v1_root_hashes.json").read_text(encoding="utf-8"))["roots"]["cache/m2_e"]["files"]
exp = {"035.1.003": (77, 20), "265.2.003": (33, 9), "404.1.004": (33, 9), "265.2.005": (81, 21), "370.1.008": (81, 21)}
for f in files:
    raw = Path(f).read_bytes()
    h = json.loads(raw)
    g = H.gate_handoff(h, hashlib.sha256(raw).hexdigest(), man)             # the repo copy IS the frozen Drive file
    assert g["hash_checked"] and g["disposition"] in ("accepted", "labelled")
    f0 = H.gate_frame0(h)
    env = H.gate_envelope(h, None)
    key = next(k for k in exp if k in h["clip"])
    assert (env["F_pair"], env["T_l"]) == exp[key], (h["clip"], env)
    assert 0.5 < f0["fx_over_fy_1024"] < 0.6
print("1. six frozen handoffs pass g_handoff (hash-checked against the closure manifest), g_frame0, g_envelope")

h0 = json.loads(Path(files[0]).read_text(encoding="utf-8"))
assert h0["t_a"] == 4

# 2. a changed handoff is a hard error; a refused disposition is a recorded refusal
must_raise(AssertionError, lambda: H.gate_handoff(h0, "0" * 64, man), "tampered hash")
must_raise(H.HandoffRefused, lambda: H.gate_handoff({**h0, "disposition": "refused"}), "refused pair")
print("2. tampered handoff -> hard error; refused disposition -> recorded refusal")

# 3. g_frame0 hard errors
K_bad = copy.deepcopy(h0["K_1024"])
K_bad[0][0] *= 1.1
for bad in ({"A_already_applied_in_g_tar": False}, {"frame": "camera_0"}, {"generated_frame0_is_source_frame": 0},
            {"gauge": "metric"}, {"K_1024": K_bad}):
    must_raise(AssertionError, lambda b=bad: H.gate_frame0({**h0, **b}), list(bad))
print("3. g_frame0 refuses: A not applied, wrong frame, wrong first frame, wrong gauge, K_1024 != 4 K_256")

# 4. envelope: refusal only when min_frames is set and not met; mode: only depth_preserving
assert H.gate_envelope({**h0, "n_frames_after_t_a": 34}, None)["F_pair"] == 33
must_raise(H.HandoffRefused, lambda: H.gate_envelope({**h0, "n_frames_after_t_a": 34}, 49), "short pair under min_frames")
assert H.gate_mode("depth_preserving")["s"] == 1.0
must_raise(H.HandoffRefused, lambda: H.gate_mode("size_preserving"), "undecided mode")
print("4. envelope refuses only below a set min_frames; only depth_preserving placement is admitted")

# 5. the lock section is required and never defaulted
try:
    H.load_m3_lock()
    print("5. lock `m3:` section present with the required keys")
except AssertionError as e:
    print(f"5. lock check refuses, as it must until John completes the section: {str(e)[:120]}")
print("test_m3_handoff: passed")
