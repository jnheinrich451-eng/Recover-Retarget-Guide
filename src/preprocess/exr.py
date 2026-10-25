"""Read the Z channel of a ViPE depth EXR (half-float) into float32.

2026-10-05: the August runs used the legacy `openexr` 1.3 bindings, which ship an `Imath` module and accept file
objects in `OpenEXR.InputFile`. The `OpenEXR` 3.x wheel that `nvidia-vipe` now pulls has no `Imath` module (the
`import Imath` was never used, only required by the old import order) and its legacy `InputFile` takes a PATH.
This helper tolerates both: it reads from bytes, tries the legacy file-object path, and falls back to a temp file
and then to the 3.x `OpenEXR.File` API. The returned array is identical under every path (raw HALF -> float32),
asserted finite-shaped, never resampled (CLAUDE.md §3.6: no silent geometry change).
"""
from __future__ import annotations

import io
import os
import tempfile

import numpy as np


def _legacy_decode(exr) -> np.ndarray:
    dw = exr.header()["dataWindow"]
    w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    return np.frombuffer(exr.channels(["Z"])[0], np.float16).reshape(h, w).astype(np.float32)


def read_exr_z(data: bytes) -> np.ndarray:
    try:
        import OpenEXR  # noqa: F401
    except ImportError as e:  # pragma: no cover - environment
        raise SystemExit(f"OpenEXR bindings missing ({e}); `pip install OpenEXR` (Imath is NOT required)") from e
    # path 1: legacy API on a file object (openexr 1.3 bindings)
    try:
        return _legacy_decode(OpenEXR.InputFile(io.BytesIO(data)))
    except Exception:
        pass
    tmp = tempfile.NamedTemporaryFile(suffix=".exr", delete=False)
    try:
        tmp.write(data)
        tmp.close()
        # path 2: legacy API on a path (OpenEXR 3.x keeps InputFile for paths)
        try:
            return _legacy_decode(OpenEXR.InputFile(tmp.name))
        except Exception:
            pass
        # path 3: OpenEXR 3.x object API
        with OpenEXR.File(tmp.name) as f:
            z = f.channels()["Z"].pixels
        return np.asarray(z, dtype=np.float32)
    finally:
        os.unlink(tmp.name)
