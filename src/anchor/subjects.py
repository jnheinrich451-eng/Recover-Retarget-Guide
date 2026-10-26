"""The target-subject registry (configs/subjects.yaml): kind, identity, detector query, arbiter label, facing offset and proxy
key per category. The two categories of the thesis are always present; others are appended by scripts/make_proxy.py."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "configs" / "subjects.yaml"
DEFAULTS = {  # the thesis's two, so that nothing changes when the registry file is absent
    "vehicle": {"kind": "rigid", "object": "a red car", "action": "driving", "query": "a car", "arbiter_label": "car",
                "facing_offset_deg": 0, "proxy_key": "5fbdb89abb315954"},
    "human": {"kind": "human", "object": "a man in a grey jacket and dark trousers", "action": "walking", "query": "a person",
              "arbiter_label": "person", "facing_offset_deg": -12, "proxy_key": "7b1379d4dfce7cad"},
}
KINDS = ("rigid", "human")
HEADER = "# Target subjects (2026-10-26). A subject is a category name with a kind, an identity (the words the proxy's candidate image and\n# the anchor prompt use), the detector query for its mask, the grader's arbiter label and, once its proxy is built, the proxy's\n# cache key. `vehicle` and `human` are the two of the thesis (their proxies of record are the release asset); any other entry\n# is appended by scripts/make_proxy.py and is experimental: the thesis's measurements say nothing about it.\n#   kind: rigid  -> SAM 3D Objects reconstructs it, Module 3 renders the object along its path (the vehicle arm)\n#   kind: human  -> SAM 3D Body reconstructs it, Module 3 holds the camera only (the human arm)\n"


LOCAL = ROOT / "configs" / "subjects.local.yaml"      # a reader's own subjects (git-ignored, so a pull never collides with them)


def load() -> dict:
    s = dict(DEFAULTS)
    for p in (PATH, LOCAL):
        if p.exists():
            d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            s.update(d.get("subjects") or {})
    return s


def get(category: str) -> dict:
    s = load()
    if category not in s:
        raise KeyError(f"unknown target subject {category!r}; known: {sorted(s)}. Register it with scripts/make_proxy.py --category {category} --object '<text>' --kind rigid|human")
    return s[category]


def kind(category: str) -> str:
    k = get(category).get("kind")
    assert k in KINDS, f"{category}: kind {k!r} is not one of {KINDS}"
    return k


def variant(category: str) -> str:
    """The reconstructor: rigid subjects go to SAM 3D Objects, humans to SAM 3D Body."""
    return "objects" if kind(category) == "rigid" else "body"


def identity(category: str) -> dict:
    s = get(category)
    return {"object": s["object"], "action": s.get("action", "moving")}


def query(category: str) -> str:
    s = get(category)
    return s.get("query") or s["object"]


def arbiter_label(category: str):
    return get(category).get("arbiter_label")


def facing_offset(category: str):
    return get(category).get("facing_offset_deg", 0)


def proxy_key(category: str):
    return get(category).get("proxy_key")


def default_name(category: str) -> str:
    """The thesis category a subject's kind corresponds to (for tables keyed on `vehicle` / `human`)."""
    return "vehicle" if kind(category) == "rigid" else "human"


def register(category: str, **fields) -> dict:
    """Add or update a subject. The thesis's two live in configs/subjects.yaml; any other subject is written to
    configs/subjects.local.yaml, which git ignores, so that registering one never blocks a pull."""
    target = PATH if category in DEFAULTS else LOCAL
    d = (yaml.safe_load(target.read_text(encoding="utf-8")) or {}) if target.exists() else {}
    subs = d.setdefault("subjects", {}) or {}
    d["subjects"] = subs
    cur = dict(subs.get(category) or load().get(category) or {})
    cur.update({k: v for k, v in fields.items() if v is not None})
    assert cur.get("kind") in KINDS, f"{category}: kind must be one of {KINDS}"
    assert cur.get("object"), f"{category}: an object text is required"
    subs[category] = cur
    head = HEADER if target is PATH else "# A reader's own target subjects, registered by scripts/make_proxy.py (git-ignored; merged over configs/subjects.yaml).\n"
    target.write_text(head + yaml.safe_dump(d, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
    return cur
