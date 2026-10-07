"""Taste traits: concrete, shopper-language descriptors ("low-rise black flared pants"), never scene labels.

Two readers:
- local (free): CLIP matches an image against the trait vocabulary (data/traits_vocab.yaml + phrases
  learned from Claude). Default path for screenshots and quiz picks.
- Claude (metered): reads a batch of images, or polishes the local read. Anything new it says is added
  to the learned vocabulary, so the free path covers more over time.

A trait lives in the profile as {id, text, embedding, enabled, sources, user_added, added_at}.
"""
import hashlib
import json
import re
import threading
import time
import uuid

import numpy as np
import yaml

from . import llm
from .config import DATA
from .embed import Embedder
from .index import Catalog
from .taste import _lookup_ci, known_brands

BANNED = re.compile(
    r"\b(\w*core|y2k|\d0s|vibe|vibes|aesthetic|aesthetics|chic|boho|grunge|preppy|coquette|cottage|streetwear|"
    r"athleisure|quiet luxury|old money|it[- ]girl|cool[- ]girl|clean[- ]girl|edgy|trendy|minimalist|maximalist|"
    r"indie|alt|retro|vintage-inspired)\b", re.I)
MAX_WORDS = 8

VOCAB_FILE = DATA / "traits_vocab.yaml"
LEARNED_FILE = DATA / "traits_vocab_learned.json"
BRAND_TRAITS_CACHE = DATA / "brand_traits.json"
IMAGE_TRAITS_CACHE = DATA / "image_traits.json"
_lock = threading.Lock()

LOCAL_PER_IMAGE = 3     # candidate phrases per image
LOCAL_MIN_Z = 1.5       # over the vocabulary's similarity distribution for that image
LOCAL_MAX = 6           # traits per batch


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().strip(".,;:!\"'").lower())


def clean(traits: list[str]) -> list[str]:
    out: list[str] = []
    for t in traits:
        t = normalize(t)
        if t and not BANNED.search(t) and len(t.split()) <= MAX_WORDS and t not in out:
            out.append(t)
    return out


# ---------- caches ----------

def cache_get(path, key):
    with _lock:
        if path.exists():
            return json.load(open(path)).get(key)
    return None


def cache_put(path, key, value) -> None:
    with _lock:
        cache = json.load(open(path)) if path.exists() else {}
        cache[key] = value
        with open(path, "w") as f:
            json.dump(cache, f, indent=1, ensure_ascii=False)


# ---------- vocabulary (free reader) ----------

_vocab: dict = {"phrases": None, "vecs": None, "stamp": None}


def _vocab_stamp() -> tuple:
    return tuple(p.stat().st_mtime if p.exists() else 0 for p in (VOCAB_FILE, LEARNED_FILE))


def vocab() -> list[str]:
    base = yaml.safe_load(open(VOCAB_FILE)) if VOCAB_FILE.exists() else []
    learned = json.load(open(LEARNED_FILE)) if LEARNED_FILE.exists() else []
    return clean([str(x) for x in base] + [str(x) for x in learned])


VOCAB_VECS_FILE = DATA / "traits_vocab_vectors.npz"   # precomputed; only new phrases are encoded at runtime


def _encode_vocab(embedder: Embedder, phrases: list[str]) -> np.ndarray:
    cached: dict[str, np.ndarray] = {}
    if VOCAB_VECS_FILE.exists():
        z = np.load(VOCAB_VECS_FILE)
        cached = dict(zip(z["phrases"].tolist(), z["vecs"]))
    missing = [p for p in phrases if p not in cached]
    if missing:
        cached.update(zip(missing, embedder.texts(missing).astype(np.float16)))
        np.savez(VOCAB_VECS_FILE, phrases=np.array(phrases), vecs=np.stack([cached[p] for p in phrases]))
    return np.stack([cached[p] for p in phrases]).astype(np.float32)


def vocab_vectors(embedder: Embedder) -> tuple[list[str], np.ndarray]:
    stamp = _vocab_stamp()
    with _lock:
        if _vocab["phrases"] is None or _vocab["stamp"] != stamp:
            phrases = vocab()
            _vocab.update(phrases=phrases, vecs=_encode_vocab(embedder, phrases), stamp=stamp)
        return _vocab["phrases"], _vocab["vecs"]


def learn(phrases: list[str]) -> int:
    """Add new phrases (e.g. from a Claude read) to the learned vocabulary. Returns how many were new."""
    known = set(vocab())
    new = [p for p in clean(phrases) if p not in known]
    if new:
        with _lock:
            learned = json.load(open(LEARNED_FILE)) if LEARNED_FILE.exists() else []
            json.dump(learned + new, open(LEARNED_FILE, "w"), indent=1, ensure_ascii=False)
    return len(new)


def from_vectors_local(embedder: Embedder, vecs: list[np.ndarray]) -> list[dict]:
    """CLIP zero-shot: for each image, the vocabulary phrases it sits closest to; merged across images.
    Returns [{"text", "images": [indices], "score"}], phrases seen in more images first. $0."""
    phrases, V = vocab_vectors(embedder)
    hits: dict[int, list] = {}
    for i, v in enumerate(vecs):
        sims = V @ np.asarray(v, dtype=np.float32)
        z = (sims - sims.mean()) / (sims.std() + 1e-6)
        for j in np.argsort(-z)[:LOCAL_PER_IMAGE]:
            if z[j] < LOCAL_MIN_Z:
                break
            h = hits.setdefault(int(j), [0.0, []])
            h[0] += float(z[j])
            h[1].append(i)
    ranked = sorted(hits.items(), key=lambda kv: (-len(kv[1][1]), -kv[1][0]))[:LOCAL_MAX]
    return [{"text": phrases[j], "images": imgs, "score": round(total, 2)} for j, (total, imgs) in ranked]


# ---------- Claude readers (metered) ----------

def from_images(jpegs: list[bytes], captions: list[str] | None = None, context: str | None = None) -> list[dict]:
    """One Claude call for the batch. Returns [{"text", "images": [indices]}]. Cached per identical batch."""
    key = hashlib.sha1("".join(hashlib.sha1(b).hexdigest() for b in jpegs).encode()).hexdigest()
    cached = cache_get(IMAGE_TRAITS_CACHE, key)
    if cached is not None:
        return cached
    kwargs = {"captions": captions}
    if context:
        kwargs["context"] = context
    found = llm.traits_from_images(jpegs, **kwargs) or []
    cleaned, seen = [], set()
    for d in found:
        t = clean([d["text"]])
        if t and t[0] not in seen:
            seen.add(t[0])
            cleaned.append({"text": t[0], "images": d.get("images", [])})
    if cleaned:
        cache_put(IMAGE_TRAITS_CACHE, key, cleaned)
        learn([d["text"] for d in cleaned])
    return cleaned


def polish(candidates: list[str], nearest_titles: list[str]) -> list[str]:
    """Text-only Claude pass over the free read: rephrase, merge, catch what the vocabulary lacks."""
    found = clean(llm.polish_traits(candidates, nearest_titles) or [])
    if found:
        learn(found)
    return found


def from_brand(catalog: Catalog, name: str) -> tuple[str | None, list[str]]:
    """Returns (catalog brand name if indexed, traits). Precomputed for every indexed and known brand;
    falls back to a Claude read only for a brand we've never seen."""
    key = catalog.brand_key(name)
    cache_key = (key or name).lower()
    cached = cache_get(BRAND_TRAITS_CACHE, cache_key)
    if cached is not None:
        return key, cached
    if key:
        rows = catalog.brand_rows[key]
        titles = [catalog.items[i]["title"] for i in rows[:80]]
        types = sorted({catalog.items[i]["product_type"] for i in rows if catalog.items[i]["product_type"]})[:25]
        found = clean(llm.traits_from_brand(key, titles, types) or [])
    else:
        _, description = _lookup_ci(known_brands(), name)
        found = clean(llm.traits_from_brand(name, [], [], description=description) or [])
    if found:
        cache_put(BRAND_TRAITS_CACHE, cache_key, found)
        learn(found)
    return key, found


# ---------- profile mutation ----------

def new_trait(text: str, embedder: Embedder, sources: list[dict], user_added: bool = False) -> dict:
    return {
        "id": uuid.uuid4().hex[:8],
        "text": text,
        "embedding": embedder.texts([text])[0].tolist(),
        "enabled": True,
        "sources": list(sources),
        "user_added": user_added,
        "added_at": time.time(),
    }


def merge(profile: dict, entries: list[tuple[str, list[dict]]], embedder: Embedder) -> list[dict]:
    """Add (text, sources) entries, merging by text so a trait seen in several places keeps every source."""
    touched = []
    by_text = {t["text"]: t for t in profile["traits"]}
    for text, sources in entries:
        cleaned = clean([text])
        if not cleaned:
            continue
        text = cleaned[0]
        if text in by_text:
            t = by_text[text]
            for s in sources:
                if s not in t["sources"]:
                    t["sources"].append(s)
        else:
            t = new_trait(text, embedder, sources)
            profile["traits"].append(t)
            by_text[text] = t
        touched.append(t)
    return touched


def remove_source(profile: dict, kind: str, ref: str) -> None:
    """Forget a screenshot, quiz pick or brand; traits left with no source are dropped (user-added ones stay)."""
    for t in profile["traits"]:
        t["sources"] = [s for s in t["sources"] if not (s["kind"] == kind and s.get("ref") == ref)]
    profile["traits"] = [t for t in profile["traits"] if t["sources"] or t.get("user_added")]


def public(trait: dict) -> dict:
    return {k: v for k, v in trait.items() if k != "embedding"}
