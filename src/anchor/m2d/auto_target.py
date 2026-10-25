"""Target text for the experiment set, by a FIXED rule from the clip's caption (plan v1: 'prompts from captions by fixed rule').

The object and the action are the proxy's own identity per category (configs/m2d.yaml: the control shows THAT geometry, so the
words must match its morphology): they never come from the caption. Only the SCENE comes from the caption, by one rule applied to
every clip and recorded next to its result:

  1. find the first motion verb of the list below in the caption (whole word); the scene is the text after it, from its first
     locative preposition when the verb takes an object ('rides a horse through ...' -> 'through ...'); the plain remainder otherwise;
  2. if no listed verb occurs, the scene starts at the first locative preposition of the list below;
  3. the scene ends at the first ", <word>" whose word ends in -ed / -ing or is one of the clause openers below, or at the first full stop;
  4. nothing else is edited: a rule failure (a verb object carried into the scene, a lost noun) is DISCLOSED, not fixed.

No clip is special-cased. The rule was written before any experiment-set anchor was generated and is not tuned on the set."""
from __future__ import annotations

import re

MOTION_VERBS = ("walks", "walk", "runs", "run", "drives", "drive", "speeds", "travels", "rides", "ride", "gallops", "traverses", "navigate",
                "navigates", "exits", "launches", "spins", "performs", "dance", "dances", "move", "moves", "clamber", "clambers", "picks",
                "swim", "swims", "skates", "clear", "clears", "carries", "jumps", "flies", "sails", "climbs", "crosses")
LOCATIVES = ("through", "along", "across", "down", "past", "around", "inside", "into", "over", "on", "in", "at", "under", "beside", "by",
             "towards", "toward", "near", "among", "between", "beneath")
CLAUSE_OPENERS = ("with", "under", "surrounded", "while", "as", "and")

def _identities():
    from src.anchor import subjects
    return {c: {"object": s["object"], "action": s.get("action", "moving")} for c, s in subjects.load().items()}


PROXY_IDENTITY = _identities()                 # the thesis's two plus any subject registered by scripts/make_proxy.py

_WB = chr(92) + "b"                                                         # the regex word boundary, spelled out so no shell ever mangles it


def _first_locative(text: str):
    hits = [(m.start(), p) for p in LOCATIVES for m in [re.search(_WB + p + _WB, text)] if m]
    return min(hits) if hits else None


def scene_from_caption(caption: str) -> dict:
    text = caption.strip()
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    verb = next((w for w in words if w.lower() in MOTION_VERBS), None)
    if verb is not None:
        m = re.search(_WB + re.escape(verb) + _WB, text)
        rest = text[m.end():].strip(); start = f"verb '{verb}'"
        if not re.match("(" + "|".join(LOCATIVES) + ")" + _WB, rest, re.I):   # rule 1b: the scene is the locative phrase after the verb;
            loc = _first_locative(rest)                                       # a verb object before it ('rides a horse', 'picks its way') is skipped
            if loc is not None:
                rest = rest[loc[0]:].strip(); start += f" -> locative '{loc[1]}'"
    else:
        loc = _first_locative(text)
        if loc is None:
            return {"scene": "", "scene_source": "none (no listed verb or locative in the caption)", "caption": caption}
        rest = text[loc[0]:].strip(); start = f"locative '{loc[1]}'"
    # rule 3: end at the first clause-opening comma or the first full stop
    end = len(rest)
    for mm in re.finditer(r",\s*([A-Za-z-]+)", rest):
        w = mm.group(1).lower()
        if w.endswith("ed") or w.endswith("ing") or w in CLAUSE_OPENERS:
            end = mm.start(); break
    dot = rest.find(".")
    if 0 <= dot < end:
        end = dot
    scene = rest[:end].strip().rstrip(",;").strip()
    return {"scene": scene, "scene_source": start, "caption": caption}


def target_for(category: str, caption: str) -> dict:
    assert category in PROXY_IDENTITY, f"no proxy identity for category {category!r}"
    s = scene_from_caption(caption)
    return {**PROXY_IDENTITY[category], "scene": s["scene"], "scene_source": s["scene_source"], "caption": caption,
            "rule": "src/anchor/m2d/auto_target.py: object/action = proxy identity per category; scene = the locative phrase after the first listed motion verb (else from the first locative), cut at the first clause-opening comma or full stop"}


def arbiter_label(category: str, caption: str) -> str:
    """ADE20K class for the grader's mask-centroid arbiter (scripts/check_m3_mask_centroid.py). Decided on the caption's SUBJECT phrase,
    the words before the first listed motion verb (reviewer 2026-10-19: 'horse-drawn carriages' in the background of 256.3.010 and the
    'rider' of a BMX must not label the subject); the whole caption only when no verb is listed. Keyword order: a ridden machine before
    its rider."""
    text = caption.strip(); words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    verb = next((w for w in words if w.lower() in MOTION_VERBS), None)
    subject = text[:re.search(_WB + re.escape(verb) + _WB, text).start()] if verb else text
    c = subject.lower()
    if category not in ("vehicle", "human"):
        from src.anchor import subjects
        return subjects.arbiter_label(category) or "person"
    if category == "vehicle":
        for kw, lab in (("carriage", "horse"), ("truck", "truck"), ("bus", "bus"), ("van", "van"), ("boat", "boat"), ("motorbike", "motorbike"), ("motorcycle", "motorbike")):
            if kw in c:
                return lab
        return "car"
    for kw, lab in (("bmx", "bicycle"), ("bike", "bicycle"), ("bicycle", "bicycle"), ("horse", "horse"), ("rider", "horse"), ("gallop", "horse"),
                    ("camel", "animal"), ("elephant", "animal"), ("goat", "animal"), ("dog", "animal"), ("puppies", "animal"), ("puppy", "animal"), ("fish", "animal")):
        if kw in c:
            return lab
    return "person"
