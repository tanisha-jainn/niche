"""Taste profile -> anchors -> scored, diversified feed."""
import json
import random
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import yaml

from . import llm
from .config import KNOWN_BRANDS_FILE, PROFILE_FILE
from .embed import Embedder
from .index import Catalog

EMPTY_PROFILE = {
    "liked_brands": [], "images": [], "pins": [], "quiz_picks": [], "traits": [],
    "saved_brands": [], "dismissed_brands": [], "discovered_brands": [], "saved_items": [],
    "sizes": {}, "currency": "USD",
    "liked_items": [], "skipped_items": [],
}


@dataclass
class Anchor:
    vec: np.ndarray
    kind: str    # "brand" | "image" | "item"
    label: str


# ---------- profile persistence (one file per visitor; `pid` comes from the browser cookie) ----------

PROFILES_DIR = PROFILE_FILE.parent / "profiles"
_PID = re.compile(r"^[a-f0-9]{32}$")


def _path(pid: str | None):
    if pid is None:
        return PROFILE_FILE                       # local scripts / CLI
    if not _PID.fullmatch(pid):
        raise ValueError("bad profile id")
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    return PROFILES_DIR / f"{pid}.json"


def load_profile(pid: str | None = None) -> dict:
    path = _path(pid)
    if path.exists():
        with open(path) as f:
            return {**json.loads(json.dumps(EMPTY_PROFILE)), **json.load(f)}
    return json.loads(json.dumps(EMPTY_PROFILE))


def save_profile(pid: str | None, profile: dict) -> None:
    with open(_path(pid), "w") as f:
        json.dump(profile, f)


def reset_profile(pid: str | None = None) -> dict:
    profile = json.loads(json.dumps(EMPTY_PROFILE))
    save_profile(pid, profile)
    return profile


def public_profile(profile: dict) -> dict:
    """Profile without raw embeddings, for the UI."""
    strip = lambda rows: [{k: v for k, v in r.items() if k != "embedding"} for r in rows]
    return {
        **profile,
        "images": strip(profile["images"]),
        "pins": strip(profile.get("pins", [])),
        "traits": strip(profile["traits"]),
    }


# ---------- known (mainstream) brands ----------

def known_brands() -> dict[str, str]:
    if not KNOWN_BRANDS_FILE.exists():
        return {}
    with open(KNOWN_BRANDS_FILE) as f:
        return {b["name"]: b["description"] for b in yaml.safe_load(f)}


def _lookup_ci(mapping: dict, name: str):
    lowered = {k.lower(): k for k in mapping}
    key = lowered.get(name.strip().lower())
    return (key, mapping[key]) if key else (None, None)


# ---------- anchors ----------

def brand_description(name: str) -> str:
    """Text used to place a brand we don't index into CLIP space."""
    _, desc = _lookup_ci(known_brands(), name)
    if desc:
        return desc
    desc = llm.describe_brand(name)
    return desc or f"clothing from the fashion brand {name}"


def build_anchors(
    profile: dict, catalog: Catalog, get_embedder: Callable[[], Embedder]
) -> tuple[list[Anchor], list[Anchor]]:
    """`get_embedder` is lazy: the CLIP model is only loaded if a brand needs a text embedding."""
    pos: list[Anchor] = []
    neg: list[Anchor] = []

    for name in profile["liked_brands"]:
        key, _ = _lookup_ci(catalog.brand_rows, name)
        if key:
            pos.append(Anchor(catalog.brand_centroid(key), "brand", key))
        else:
            pos.append(Anchor(get_embedder().texts([brand_description(name)])[0], "brand", name))

    for im in profile["images"]:
        pos.append(Anchor(np.asarray(im["embedding"], dtype=np.float32), "image", im["name"]))

    for pid in profile["liked_items"]:
        i = catalog.by_id.get(pid)
        if i is not None:
            it = catalog.items[i]
            pos.append(Anchor(catalog.E[i], "item", f"{it['brand']} · {it['title']}"))

    for pid in profile["skipped_items"]:
        i = catalog.by_id.get(pid)
        if i is not None:
            neg.append(Anchor(catalog.E[i], "item", catalog.items[i]["title"]))

    return pos, neg


# ---------- scoring ----------

def _zscore(sims: np.ndarray) -> np.ndarray:
    # Text->image and image->image cosine similarities live on different scales.
    # Standardising each anchor's similarity distribution over the catalog makes them comparable.
    sd = sims.std()
    return (sims - sims.mean()) / (sd if sd > 1e-6 else 1.0)


def score(catalog: Catalog, pos: list[Anchor], neg: list[Anchor]) -> tuple[np.ndarray, np.ndarray]:
    Z = np.stack([_zscore(catalog.E @ a.vec) for a in pos])          # (anchors, items)
    best = Z.argmax(axis=0)                                            # which anchor explains each item
    s = 0.6 * Z.max(axis=0) + 0.4 * Z.mean(axis=0)                     # max: multi-cluster taste; mean: overall fit
    if neg:
        Zn = np.stack([_zscore(catalog.E @ a.vec) for a in neg]).max(axis=0)
        s -= 0.5 * np.clip(Zn - 1.5, 0, None)                          # only punish look-alikes of skipped items
    return s, best


def rank(
    catalog: Catalog,
    scores: np.ndarray,
    exclude_brands: set[str],
    exclude_ids: set[str],
    limit: int = 48,
    per_brand: int = 3,
    dedupe: float = 0.95,
) -> list[int]:
    """Greedy top-k with a per-brand cap and near-duplicate suppression (same piece, other colour)."""
    excluded = {b.lower() for b in exclude_brands}
    chosen: list[int] = []
    counts: Counter = Counter()
    chosen_vecs: list[np.ndarray] = []
    for i in np.argsort(-scores):
        it = catalog.items[i]
        if it["brand"].lower() in excluded or it["id"] in exclude_ids or not it["available"]:
            continue
        if counts[it["brand"]] >= per_brand:
            continue
        v = catalog.E[i]
        if chosen_vecs and float((np.stack(chosen_vecs) @ v).max()) > dedupe:
            continue
        chosen.append(int(i))
        counts[it["brand"]] += 1
        chosen_vecs.append(v)
        if len(chosen) >= limit:
            break
    return chosen


def cold_start(catalog: Catalog, exclude_ids: set[str], limit: int = 48) -> list[int]:
    """No taste signal yet: one in-stock piece per brand, shuffled."""
    picks = []
    for rows in catalog.brand_rows.values():
        pool = [i for i in rows if catalog.items[i]["available"] and catalog.items[i]["id"] not in exclude_ids]
        if pool:
            picks.append(random.choice(pool))
    random.shuffle(picks)
    return picks[:limit]


def build_feed(
    profile: dict, catalog: Catalog, get_embedder: Callable[[], Embedder], limit: int = 48, explore: bool = False
) -> dict:
    pos, neg = build_anchors(profile, catalog, get_embedder)
    exclude_ids = set(profile["liked_items"]) | set(profile["skipped_items"])
    if not pos:
        rows = cold_start(catalog, exclude_ids, limit)
        items = [{**catalog.items[i], "because": None} for i in rows]
        return {"items": items, "cold_start": True, "brands": len({it["brand"] for it in items})}

    scores, best = score(catalog, pos, neg)
    rows = rank(catalog, scores, set(profile["liked_brands"]), exclude_ids, limit, per_brand=1 if explore else 3)
    items = []
    for i in rows:
        a = pos[best[i]]
        items.append({**catalog.items[i], "score": round(float(scores[i]), 3),
                      "because": {"kind": a.kind, "label": a.label}})
    return {"items": items, "cold_start": False, "brands": len({it["brand"] for it in items})}
