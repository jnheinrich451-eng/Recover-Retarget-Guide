"""Routing, cache key and the two records (SKILL §1, §8).

R1 `proxy.npz` is a NORMALISED shape (the object C1 places); R2 `anchor.npz` is METRIC in Stage
A's camera-t_a frame. Two files, never one field with two meanings. The cache key carries the
human's pick (selected_k, facing, image sha256): re-picking produces a new proxy, never a reuse.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

PROXY_SCHEMA = "m2b2_proxy_v1"
ANCHOR_SCHEMA = "m2b2_anchor_v1"
EXCLUDED_TARGETS = ("rider", "figurine")    # rider: composite; figurine: 0/6 strict-profile candidates (2026-09-17)


class TargetExcluded(ValueError):
    """The category is out of v1 target scope (rider: human + animal composite)."""


class VariantMismatch(ValueError):
    """R2's routed variant differs from the one recorded at R1 for the category."""


def route(category: str, cfg: dict) -> str:
    if category in EXCLUDED_TARGETS:
        why = {"rider": "human + animal composite (2026-09-14)",
               "figurine": "FLUX.1 dev gave 0 strict side profiles in 6 candidates — a category prior, not a sampling "
                           "shortfall; the principled fix is a measured f_can from the mesh's symmetry plane (2026-09-17)"}
        raise TargetExcluded(f"{category!r} targets are excluded from v1 scope: {why.get(category, '')}; sources are unaffected")
    table = cfg["m2b2"]["routing"]
    if category not in table:
        from src.anchor import subjects                                   # a registered subject (configs/subjects.yaml): by its kind
        try:
            v = subjects.variant(category)
        except KeyError:
            raise KeyError(f"no routing for category {category!r}; known: {sorted(table)} and configs/subjects.yaml")
        assert v in ("objects", "body"), v
        return v
    v = table[category]
    assert v in ("objects", "body"), v
    return v


def cache_key(category: str, prompt_template: str, seed_set, variant: str, adapter_version: str,
              model_version: str, selected_k: int, facing: str, image_sha256: str) -> str:
    """No clip enters: the proxy is per category (test 8)."""
    blob = json.dumps({"category": category, "prompt_template": prompt_template, "seed_set": [int(s) for s in seed_set],
                       "variant": variant, "adapter_version": adapter_version, "model_version": model_version,
                       "selected_k": int(selected_k), "facing": facing, "image_sha256": image_sha256}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


assert "clip" not in inspect.signature(cache_key).parameters


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_proxy(out_dir: Path, V_norm, F, B_tar, facing: str, o_p, norm_scale: float, qc: dict):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    V = np.asarray(V_norm, np.float32)
    assert facing in ("right", "left") and np.isfinite(V).all()
    assert abs(np.linalg.norm(V.astype(np.float64), axis=1).max() - 1.0) < 1e-4, "proxy must be normalised (max|v| = 1)"
    np.savez_compressed(out_dir / "proxy.npz", schema=PROXY_SCHEMA, vertices=V, faces=np.asarray(F, np.int32),
                        B_tar=np.asarray(B_tar, np.float32), facing=facing, o_p=np.asarray(o_p, np.float32),
                        norm_scale=np.float32(norm_scale))
    qc = {"schema": PROXY_SCHEMA, **qc}
    (out_dir / "qc_b2.json").write_text(json.dumps(qc, indent=1, default=_json_default), encoding="utf-8")
    return out_dir / "proxy.npz"


def read_proxy(out_dir: Path, expected_variant: str, adapter_version: str) -> dict:
    """Refuses a record whose variant or adapter version differs (test 9) or that lacks a facing."""
    out_dir = Path(out_dir)
    qc = json.loads((out_dir / "qc_b2.json").read_text(encoding="utf-8"))
    with np.load(out_dir / "proxy.npz") as d:
        assert str(d["schema"]) == PROXY_SCHEMA and qc.get("schema") == PROXY_SCHEMA, "proxy schema mismatch"
        if qc.get("variant") != expected_variant:
            raise VariantMismatch(f"proxy recorded with variant {qc.get('variant')!r}, R2 routed {expected_variant!r}")
        if qc.get("adapter_version") != adapter_version:
            raise ValueError(f"proxy adapter_version {qc.get('adapter_version')!r} != {adapter_version!r}; regenerate")
        facing = str(d["facing"])
        assert facing in ("right", "left"), facing
        return {"vertices": d["vertices"].astype(np.float64), "faces": d["faces"].astype(np.int64),
                "B_tar": d["B_tar"].astype(np.float64), "facing": facing, "o_p": d["o_p"].astype(np.float64),
                "norm_scale": float(d["norm_scale"]), "qc": qc}


def write_anchor(out_dir: Path, X_cv, F, K_a, t_a: int, qc: dict, gaussians=None):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    X = np.asarray(X_cv, np.float32)
    assert np.isfinite(X).all() and (X[:, 2] > 0).mean() > 0.5, "anchor must sit in front of the camera"
    arrays = {"schema": ANCHOR_SCHEMA, "vertices": X, "faces": np.asarray(F, np.int32),
              "K_a": np.asarray(K_a, np.float32), "t_a": np.int64(t_a)}
    if gaussians is not None:
        arrays.update({f"gs_{k}": np.asarray(v, np.float32) for k, v in gaussians.items()})
    np.savez_compressed(out_dir / "anchor.npz", **arrays)
    (out_dir / "qc_b2.json").write_text(json.dumps({"schema": ANCHOR_SCHEMA, **qc}, indent=1, default=_json_default), encoding="utf-8")
    return out_dir / "anchor.npz"


def _json_default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def promote(review_root: Path, cache_root: Path, category: str, key: str, prompt_template: str) -> Path:
    """Move <review_root>/<category>/<key> to <cache_root>/<category>/<key> AFTER proving that the
    key re-derives from the record's own fields with the deployed code (review 2026-09-14: a proxy
    promoted under a key the code will not look up is exactly the stale-proxy failure)."""
    import shutil
    src = Path(review_root) / category / key
    qc = json.loads((src / "qc_b2.json").read_text(encoding="utf-8"))
    rederived = cache_key(qc["category"], prompt_template, qc["seeds"], qc["variant"], qc["adapter_version"],
                          qc["model_version"], qc["selected_k"], qc["facing"], qc["image_sha256"])
    if rederived != key or qc.get("cache_key") != key:
        raise ValueError(f"key {key} does not re-derive from its record ({rederived}); regenerate with --commit-cache")
    assert qc["category"] == category
    dst = Path(cache_root) / category / key
    if dst.exists():
        raise FileExistsError(f"{dst} already exists; refusing to overwrite a cached proxy")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return dst


def select_key(root, key=None, proxy_gen=None):
    """Pick a proxy cache key under <root>: explicit key, or the one whose candidate image is
    <proxy_gen>.png (2026-09-15: a category can now hold several proxies, e.g. the rejected human s0
    and its strict replacement s2), or the only one present."""
    import json as _json
    root = Path(root)
    keys = sorted(d.name for d in root.iterdir() if d.is_dir()) if root.exists() else []
    if key:
        assert key in keys, f"key {key} not under {root}; found {keys}"
        return key
    if proxy_gen:
        hits = []
        for k in keys:
            q = _json.loads((root / k / "qc_b2.json").read_text(encoding="utf-8"))
            if Path(str(q.get("image_path", ""))).stem == proxy_gen:
                hits.append(k)
        assert len(hits) == 1, f"candidate {proxy_gen}: {len(hits)} proxies under {root} ({hits}); give --key"
        return hits[0]
    assert len(keys) == 1, f"{len(keys)} proxies under {root} ({keys}); give --proxy-gen <candidate name> or --key"
    return keys[0]
