"""Where the source video is on THIS machine. A bundle's provenance records the absolute path the clip had when Stage A ran; on another
VM (or after a re-download to a different folder) that path is stale. Resolution order: the recorded path; <MT_DATA>/videos/<clip>.mp4;
<MT_DATA>/videos/<pool>/<clip>.mp4; <MT_DATA>/*/<clip>.mp4; <MT_DATA>/*/*/<clip>.mp4. Exactly one match or an error that lists what was tried."""
from __future__ import annotations

import os
from pathlib import Path


def resolve_clip_path(recorded, clip: str, data_dir: str | None = None) -> Path:
    tried = []
    if recorded:
        p = Path(str(recorded))
        if p.exists():
            return p
        tried.append(str(p))
    data = Path(data_dir or os.environ.get("MT_DATA", "/content/data"))
    for pat in (f"videos/{clip}.mp4", f"videos/*/{clip}.mp4", f"*/{clip}.mp4", f"*/*/{clip}.mp4"):
        hits = sorted(data.glob(pat)); tried.append(str(data / pat))
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise FileNotFoundError(f"source video for {clip} is ambiguous under {data}: {hits}")
    raise FileNotFoundError(f"source video for {clip} not found; tried: " + "; ".join(tried))
