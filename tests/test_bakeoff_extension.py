"""CPU test for the A.5 calibration-record EXTENSION path (scripts/run_bakeoff.py, 2026-10-04).
No external model: tests merge_record's rules (archived entries never dropped or silently replaced) and the CLI parse.

Run:  python tests/test_bakeoff_extension.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_bakeoff import merge_record  # noqa: E402

ARCHIVED = {"000000000035.1.003": {"sim3_s": 1.1, "sigma_abs": [1.0, 1.01]},
            "rallye": {"sim3_s": 0.9, "sigma_abs": [1.0, None]},
            "000000000265.2.003": {"sim3_s": 1.0, "sigma_abs": [1.0, 1.0]}}
NEW = {"000000000404.1.004": {"sim3_s": 1.2, "sigma_abs": [1.0, 1.02], "extension_run": True}}

with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "calibration_record.json"
    p.write_text(json.dumps(ARCHIVED), encoding="utf-8")

    # 1. add a new clip: all archived entries survive byte-for-byte, new one appended
    m = merge_record(p, NEW)
    assert set(m) == set(ARCHIVED) | set(NEW)
    for k, v in ARCHIVED.items():
        assert m[k] == v, k
    assert m["000000000404.1.004"]["extension_run"] is True

    # 2. clash without force refuses (SystemExit), record untouched on disk
    try:
        merge_record(p, {"rallye": {"sim3_s": 5.0}})
        raise AssertionError("clash must refuse")
    except SystemExit as e:
        assert "rallye" in str(e)
    assert json.loads(p.read_text(encoding="utf-8")) == ARCHIVED

    # 3. clash with force replaces only that clip
    m = merge_record(p, {"rallye": {"sim3_s": 5.0}}, force=True)
    assert m["rallye"] == {"sim3_s": 5.0} and m["000000000035.1.003"] == ARCHIVED["000000000035.1.003"]

    # 4. no record on disk: merge is identity
    p.unlink()
    assert merge_record(p, NEW) == NEW

# 5. CLI: default = calibration set (no --clips); extension parses a list
import argparse  # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument("--clips", nargs="+", default=None)
ap.add_argument("--force-record", action="store_true")
assert ap.parse_args([]).clips is None
a = ap.parse_args(["--clips", "000000000404.1.004", "000000000370.1.008"])
assert a.clips == ["000000000404.1.004", "000000000370.1.008"] and a.force_record is False

print("test_bakeoff_extension: 5/5 passed")
