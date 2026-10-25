"""Config loader for Stage A (SKILL §1.4).

Single source of truth: modules take values from here, never from literals.
Supports ${a.b} interpolation, flags unfilled "<...>" placeholder fields, and
computes the content-addressing hash used by bundle caching (SKILL §5) so a
threshold change can never silently reuse a stale cache.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import yaml

_INTERP = re.compile(r"\$\{([A-Za-z0-9_.]+)\}")
_PLACEHOLDER = re.compile(r"^<.*>$")

# Sections whose change must invalidate cached bundles.
_HASHED_SECTIONS = ("d4rt", "video", "thresholds")

# Content-addressing scope: ONLY the thresholds that shape Stage A bundle
# content enter Stage A's hash. Stage B gate values live in the same lock but
# cannot affect Stage A geometry — without this allowlist, adding Stage B
# entries spuriously invalidated every bundle (measured 2026-08-08). The
# filtered subset reproduces pre-Stage-B hashes exactly (same keys, values).
_STAGE_A_THRESHOLD_KEYS = (
    "tau_v", "tau_c", "tau_g_rel", "a_g2_min_inlier_frac", "tau_g_px",
    "tau_g2b_px", "a_cov_min_tracks", "a_cov_min_frac", "anchor_stride")


def _lookup(root: dict, dotted: str):
    node = root
    for part in dotted.split("."):
        node = node[part]
    return node


def _interpolate(node, root):
    if isinstance(node, dict):
        return {k: _interpolate(v, root) for k, v in node.items()}
    if isinstance(node, list):
        return [_interpolate(v, root) for v in node]
    if isinstance(node, str):
        prev = None
        while prev != node:
            prev = node
            node = _INTERP.sub(lambda m: str(_lookup(root, m.group(1))), node)
        return node
    return node


def unfilled_fields(cfg: dict, prefix: str = "") -> list[str]:
    """Dotted paths of fields still holding '<...>' placeholders."""
    out: list[str] = []
    for k, v in cfg.items():
        if k.startswith("_"):
            continue
        path = f"{prefix}{k}"
        if isinstance(v, dict):
            out += unfilled_fields(v, path + ".")
        elif isinstance(v, str) and _PLACEHOLDER.match(v.strip()):
            out.append(path)
    return out


def apply_root_override(cfg: dict) -> dict | None:
    """MT_ROOT / MT_DATA (environment) re-root every `paths.*` entry that sits under the configured drive_root / data_dir, so the
    same configs run on Colab (Drive) and on a local machine. Paths are outside the config hash, so caches stay valid."""
    import os
    paths = cfg.get("paths")
    if not isinstance(paths, dict):
        return None
    out = {}
    for env, key in (("MT_ROOT", "drive_root"), ("MT_DATA", "data_dir")):
        new = os.environ.get(env)
        old = paths.get(key)
        if not new or not isinstance(old, str) or new == old:
            continue
        new = str(Path(new))
        for k, v in list(paths.items()):
            if isinstance(v, str) and (v == old or v.startswith(old.rstrip("/") + "/")):
                paths[k] = new + v[len(old.rstrip("/")):]
        out[key] = {"from": old, "to": new}
    return out or None


def config_hash(cfg: dict) -> str:
    """sha256 (truncated) over the geometry/threshold-affecting subset."""
    subset = {k: cfg.get(k) for k in _HASHED_SECTIONS}
    th = subset.get("thresholds")
    if isinstance(th, dict):
        subset["thresholds"] = {k: th[k] for k in _STAGE_A_THRESHOLD_KEYS
                                if k in th}
    blob = json.dumps(subset, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def load_config(path: str | Path = "configs/stage_a.yaml",
                require_filled: tuple[str, ...] = ()) -> dict:
    """Load, interpolate, and validate the Stage A config.

    Gate thresholds are governed by configs/calibration.lock.yaml (READ-ONLY
    to Claude Code, CLAUDE.md §4): its `thresholds` section is merged over the
    stage config's, and its content participates in the config hash — so a
    human threshold change invalidates cached bundles exactly like any other
    geometry-affecting config change.

    require_filled: dotted prefixes that must not contain '<...>' placeholders
    for the calling step — e.g. the probe passes ("d4rt",); the full Stage A
    run passes ("repo", "d4rt"). Unrelated placeholders are tolerated but
    reported under cfg["_meta"]["unfilled"].
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = _interpolate(copy.deepcopy(raw), raw)
    root_override = apply_root_override(cfg)

    lock_path = path.parent / "calibration.lock.yaml"
    if not lock_path.exists():
        raise FileNotFoundError(
            f"{lock_path} missing — gate thresholds live there (CLAUDE.md §4)")
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
    cfg.setdefault("thresholds", {}).update(lock.get("thresholds", {}))
    # The lock's `m2:` section (since 2026-10-17) holds the Module 2 thresholds as scalars beside its `selection:` map; the Module 2
    # runners read them through cfg["thresholds"] as they always did. Scalars merge here; maps (selection) stay under lock["m2"] for the
    # readers that address them directly. A key present in both sections is a lock error, not a silent override.
    m2 = lock.get("m2") or {}
    both = sorted(k for k, v in m2.items() if not isinstance(v, dict) and k in (lock.get("thresholds") or {}))
    assert not both, f"{lock_path}: keys in both `thresholds:` and `m2:` ({both}); the lock must hold each key once"
    cfg["thresholds"].update({k: v for k, v in m2.items() if not isinstance(v, dict)})

    missing = unfilled_fields(cfg)
    blocked = [m for m in missing if m.startswith(tuple(require_filled))] if require_filled else []
    if blocked:
        raise ValueError(
            f"{path} has unfilled fields required for this step: "
            + ", ".join(blocked)
            + "  — fill them once (human decision, SKILL §1.4) and re-run."
        )
    # Hash is computed on the MERGED config, so the lock file's semantic
    # content (not its comments) is part of the hash.
    cfg["_meta"] = {"config_hash": config_hash(cfg), "unfilled": missing,
                    "path": str(path), "lock_path": str(lock_path), "root_override": root_override}
    return cfg


def load_stage_a5(path_a5: str | Path = "configs/stage_a5.yaml",
                  path_a: str | Path = "configs/stage_a.yaml") -> dict:
    """Stage A.5 config: the Stage A config (with lock) plus the A.5 sections,
    with ${paths.*} in stage_a5.yaml interpolating against Stage A's paths.
    A.5 is read-only w.r.t. Stage A — its keys never enter the Stage A hash."""
    cfg = load_config(path_a)
    raw5 = yaml.safe_load(Path(path_a5).read_text(encoding="utf-8"))
    root = copy.deepcopy(raw5)
    root["paths"] = {**cfg["paths"], **raw5.get("paths", {})}
    a5 = _interpolate(copy.deepcopy(raw5), root)
    for key in ("vipe", "clips", "align"):
        cfg[key] = a5[key]
    cfg["paths"]["out_dir"] = a5["paths"]["out_dir"]
    return cfg
