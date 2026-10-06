"""Niche API: relevance-first search over independent brands, one storefront, optional personal lens.
Run: uvicorn app.server:app --reload"""
import io
import re
import threading
import time
import uuid
from typing import Literal
from xml.etree import ElementTree

import httpx
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from . import curate, llm, quiz, recommend, search, taste, traits
from .config import UPLOADS, WEB
from .crawl import UA
from .embed import Embedder
from .index import Catalog

app = FastAPI(title="Niche")
catalog = Catalog()
UPLOADS.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOADS), name="uploads")

_embedder: Embedder | None = None
_search: search.Search | None = None
_lock = threading.Lock()


def embedder() -> Embedder:
    global _embedder
    with _lock:
        if _embedder is None:
            _embedder = Embedder()
    return _embedder


def engine() -> search.Search:
    global _search
    if _search is None:
        e = embedder()
        with _lock:
            if _search is None:
                _search = search.Search(catalog, e)
    return _search


def _require_llm() -> None:
    if not llm.available():
        raise HTTPException(503, "This needs Claude - put ANTHROPIC_API_KEY in .env and restart.")
    if not llm.budget_left():
        u = llm.usage_summary()
        raise HTTPException(402, f"Claude budget reached (${u['cost_usd']:.2f} of ${u['budget_usd']:.2f}).")


def taste_vector(p: dict) -> np.ndarray | None:
    """Mean of everything the user has shown us they like: screenshots, pins, quiz picks."""
    vecs = [np.asarray(im["embedding"], dtype=np.float32) for im in p["images"]]
    vecs += [np.asarray(pin["embedding"], dtype=np.float32) for pin in p.get("pins", [])]
    vecs += [catalog.E[catalog.by_id[pid]] for pid in p.get("quiz_picks", []) if pid in catalog.by_id]
    if not vecs:
        return None
    v = np.mean(vecs, axis=0)
    return v / (np.linalg.norm(v) or 1.0)


# ---------- pages & meta ----------

@app.get("/")
def home():
    return FileResponse(WEB / "index.html")


@app.get("/curate")
def curate_page():
    return FileResponse(WEB / "curate.html")


@app.get("/api/curate")
def curate_list():
    return {"brands": curate.overview(), "summary": curate.summary(),
            "live_brands": len(catalog.brand_rows), "excluded": catalog.excluded_brands}


class CurateIn(BaseModel):
    key: str
    decision: str | None = None     # keep | drop | null (clear)


@app.post("/api/curate")
def curate_decide(body: CurateIn):
    curate.decide(body.key.lower(), body.decision)
    return {"summary": curate.summary()}


@app.post("/api/curate/apply")
def curate_apply():
    """Rebuild the live index with the current tiers + decisions (a few seconds)."""
    global catalog, _search
    with _lock:
        catalog = Catalog()
        _search = None
    return {"live_brands": len(catalog.brand_rows), "items": len(catalog.items), "excluded": catalog.excluded_brands}


@app.get("/api/meta")
def meta():
    cats = engine().cats
    return {
        "stats": {"items": len(catalog.items), "brands": len(catalog.brand_rows),
                  "sized": sum(1 for it in catalog.items if it.get("sizes_offered"))},
        "categories": list(cats["names"]) if cats else [],
        "fx": search.FX,
        "llm": llm.available(),
        "usage": llm.usage_summary(),
        "catalog_brands": sorted(catalog.brand_rows),
        "known_brands": sorted(taste.known_brands()),
    }


@app.get("/api/usage")
def usage():
    return llm.usage_summary()


@app.get("/api/profile")
def profile():
    return {"profile": taste.public_profile(taste.load_profile())}


class SettingsIn(BaseModel):
    sizes: dict[str, str] | None = None
    currency: str | None = None


@app.post("/api/profile/settings")
def settings(body: SettingsIn):
    p = taste.load_profile()
    if body.sizes is not None:
        p["sizes"] = {k: v for k, v in body.sizes.items() if k in ("tops", "bottoms", "dresses") and v}
    if body.currency:
        p["currency"] = body.currency.upper()
    taste.save_profile(p)
    return taste.public_profile(p)


@app.post("/api/profile/reset")
def reset():
    for im in taste.load_profile()["images"]:
        (UPLOADS / f"{im['id']}.jpg").unlink(missing_ok=True)
    return taste.public_profile(taste.reset_profile())


# ---------- search ----------

@app.get("/api/search")
def do_search(q: str = "", category: str | None = None, size: bool = False, unknown: bool = True,
              price_min: float | None = None, price_max: float | None = None, personalize: bool = False,
              sort: str = "relevance", page: int = 0, brand: str | None = None):
    q = q.strip()
    if not q and not brand and not category:
        raise HTTPException(400, "Type what you're looking for.")
    p = taste.load_profile()
    fx = search.FX.get(p.get("currency", "USD"), 1.0)      # user's currency -> USD, which prices are indexed in
    result = engine().run(
        q or (category or ""), category=category, sizes=p.get("sizes") or None, size_filter=size,
        include_unknown_size=unknown, min_usd=price_min * fx if price_min is not None else None,
        max_usd=price_max * fx if price_max is not None else None,
        taste=taste_vector(p) if personalize else None,
        brands={brand} if brand else None, sort=sort, page=max(0, page),
    )
    saved = set(p.get("saved_items", []))
    for it in result["items"]:
        it["saved"] = it["id"] in saved
    result["personalized"] = bool(personalize and taste_vector(p) is not None)
    return result


@app.get("/api/items")
def item_detail(id: str):
    it = catalog.item(id)
    if it is None:
        raise HTTPException(404, "unknown item")
    p = taste.load_profile()
    eng = engine()
    i = catalog.by_id[id]
    detail = eng.result(i, 0.0, 0.0, {"tokens": []})
    detail.update({"description": it.get("description") or "", "sizes": it.get("sizes") or [],
                   "tags": (it.get("tags") or [])[:8], "domain": it["domain"],
                   "saved": id in set(p.get("saved_items", []))})
    return {"item": detail, "similar": eng.similar(id, limit=12), "more": eng.similar(id, limit=8, same_brand=True)}


class ItemFeedback(BaseModel):
    id: str
    action: Literal["save", "unsave"]
    via: str | None = None


@app.post("/api/items/feedback")
def item_feedback(body: ItemFeedback):
    it = catalog.item(body.id)
    if it is None:
        raise HTTPException(404, "unknown item")
    p = taste.load_profile()
    p["saved_items"] = [x for x in p.get("saved_items", []) if x != body.id]
    if body.action == "save":
        p["saved_items"].append(body.id)
        _discover(p, it["brand"], body.via or "saved a piece")
    taste.save_profile(p)
    return {"saved_items": p["saved_items"]}


@app.get("/api/saved/items")
def saved_items():
    p = taste.load_profile()
    eng = engine()
    out = []
    for pid in reversed(p.get("saved_items", [])):
        i = catalog.by_id.get(pid)
        if i is not None:
            r = eng.result(i, 0.0, 0.0, {"tokens": []})
            r["saved"] = True
            out.append(r)
    return {"items": out}


def _discover(p: dict, brand: str, via: str) -> None:
    found = p.setdefault("discovered_brands", [])
    for d in found:
        if d["brand"] == brand:
            d["count"] = d.get("count", 1) + 1
            d["last"] = time.time()
            return
    found.append({"brand": brand, "via": via, "first": time.time(), "last": time.time(), "count": 1})


class DiscoverIn(BaseModel):
    brand: str
    via: str | None = None


@app.post("/api/brands/discovered")
def discovered_add(body: DiscoverIn):
    if body.brand not in catalog.brand_rows:
        raise HTTPException(404, "unknown brand")
    p = taste.load_profile()
    _discover(p, body.brand, body.via or "visited")
    taste.save_profile(p)
    return {"discovered": p["discovered_brands"]}


@app.get("/api/brands/discovered")
def discovered_list():
    p = taste.load_profile()
    out = []
    for d in sorted(p.get("discovered_brands", []), key=lambda d: -d["last"]):
        rows = [i for i in catalog.brand_rows.get(d["brand"], []) if catalog.items[i]["available"]]
        meta_ = catalog.brand_meta.get(d["brand"], {})
        out.append({**d, "domain": meta_.get("domain"), "tags": meta_.get("tags", []), "count_items": len(rows),
                    "preview": [catalog.items[i]["image"] for i in rows[:4]],
                    "traits": traits.cache_get(traits.BRAND_TRAITS_CACHE, d["brand"].lower()) or []})
    return {"brands": out}


# ---------- taste inputs: screenshots, pins, quiz ----------

def _ingest_images(p: dict, blobs: list[tuple[str, bytes]], kind: str = "image", extra: dict | None = None) -> list[dict]:
    """Embed images, store thumbnails, read traits for free, attach as anchors."""
    entries = []
    for name, raw in blobs:
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            continue
        vec = embedder().images([img])[0]
        image_id = uuid.uuid4().hex[:10]
        img.thumbnail((768, 768))
        img.save(UPLOADS / f"{image_id}.jpg", "JPEG", quality=85)
        entries.append({"id": image_id, "name": name, "url": f"/uploads/{image_id}.jpg", "traits": [],
                        "embedding": vec.tolist(), "added_at": time.time(), **(extra or {})})
    if not entries:
        return []
    found = traits.from_vectors_local(embedder(), [np.asarray(e["embedding"], dtype=np.float32) for e in entries])
    merged = []
    for d in found:
        shown = [entries[i] for i in d["images"]] or entries
        for e in shown:
            e["traits"].append(d["text"])
        merged.append((d["text"], [{"kind": kind, "ref": e["id"], "url": e["url"]} for e in shown]))
    traits.merge(p, merged, embedder())
    return entries


@app.post("/api/profile/images")
async def add_images(files: list[UploadFile] = File(...)):
    p = taste.load_profile()
    blobs = []
    for f in files:
        raw = await f.read()
        if len(raw) > 15_000_000:
            raise HTTPException(413, f"{f.filename} is too large")
        blobs.append((f.filename or "image", raw))
    entries = _ingest_images(p, blobs)
    if not entries:
        raise HTTPException(400, "No readable images.")
    p["images"].extend(entries)
    taste.save_profile(p)
    return {"added": [{k: v for k, v in e.items() if k != "embedding"} for e in entries],
            "profile": taste.public_profile(p)}


class UrlsIn(BaseModel):
    urls: list[str]


def _fetch_images(urls: list[str], limit: int = 40) -> list[tuple[str, bytes]]:
    blobs = []
    with httpx.Client(headers={"User-Agent": UA}, follow_redirects=True, timeout=15) as client:
        for u in urls[:limit]:
            try:
                r = client.get(u)
                if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/"):
                    blobs.append((u, r.content))
            except httpx.HTTPError:
                continue
    return blobs


@app.post("/api/profile/image_urls")
def add_image_urls(body: UrlsIn):
    """Dragging an image from another site onto the page hands us its URL."""
    p = taste.load_profile()
    entries = _ingest_images(p, _fetch_images(body.urls))
    if not entries:
        raise HTTPException(400, "Couldn't fetch any images from those links.")
    p["images"].extend(entries)
    taste.save_profile(p)
    return {"added": len(entries), "profile": taste.public_profile(p)}


class PinterestIn(BaseModel):
    url: str


@app.post("/api/profile/pinterest")
def add_pinterest(body: PinterestIn):
    """A public board's RSS feed - no Pinterest developer approval needed."""
    m = re.search(r"pinterest\.[a-z.]+/([^/?#]+)/([^/?#]+)", body.url)
    if not m:
        raise HTTPException(400, "Paste a board link like https://www.pinterest.com/you/board-name/")
    user, board = m.group(1), m.group(2)
    rss = f"https://www.pinterest.com/{user}/{board}.rss"
    try:
        with httpx.Client(headers={"User-Agent": UA}, follow_redirects=True, timeout=20) as client:
            r = client.get(rss)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Couldn't reach Pinterest ({e.__class__.__name__}).")
    if r.status_code != 200:
        raise HTTPException(502, f"Pinterest answered {r.status_code} for that board's feed - is the board public?")
    try:
        root = ElementTree.fromstring(r.text)
    except ElementTree.ParseError:
        raise HTTPException(502, "Pinterest didn't return a readable feed for that board.")
    pins = []
    for item in root.iter("item"):
        desc = item.findtext("description") or ""
        img = re.search(r'<img[^>]+src="([^"]+)"', desc)
        if img:
            src = img.group(1).replace("/236x/", "/564x/")
            pins.append((src, item.findtext("link") or src))
    if not pins:
        raise HTTPException(404, "That board's feed has no pins we can read.")
    p = taste.load_profile()
    have = {pin.get("source") for pin in p.get("pins", [])}
    url_to_link = {src: link for src, link in pins if src not in have}
    entries = _ingest_images(p, _fetch_images(list(url_to_link)), kind="pin", extra={"board": f"{user}/{board}"})
    for e in entries:                       # _ingest_images keeps the fetched URL as the entry's name
        e["source"] = e["name"]
        e["link"] = url_to_link.get(e["name"], e["name"])
    p.setdefault("pins", []).extend(entries)
    taste.save_profile(p)
    return {"added": len(entries), "board": f"{user}/{board}", "profile": taste.public_profile(p)}


@app.delete("/api/profile/images/{image_id}")
def remove_image(image_id: str):
    p = taste.load_profile()
    p["images"] = [im for im in p["images"] if im["id"] != image_id]
    p["pins"] = [pin for pin in p.get("pins", []) if pin["id"] != image_id]
    traits.remove_source(p, "image", image_id)
    traits.remove_source(p, "pin", image_id)
    taste.save_profile(p)
    (UPLOADS / f"{image_id}.jpg").unlink(missing_ok=True)
    return taste.public_profile(p)


@app.get("/api/quiz")
def quiz_cards(n: int = 20):
    return {"items": quiz.spread(catalog, n=max(8, min(n, 40)))}


class QuizIn(BaseModel):
    picked: list[str]


@app.post("/api/quiz")
def quiz_done(body: QuizIn):
    picks = [it for pid in body.picked if (it := catalog.item(pid))]
    if not picks:
        raise HTTPException(400, "Pick at least one piece.")
    p = taste.load_profile()
    p["quiz_picks"] = list(dict.fromkeys(p.get("quiz_picks", []) + [it["id"] for it in picks]))
    found = traits.from_vectors_local(embedder(), [catalog.E[catalog.by_id[it["id"]]] for it in picks])
    merged = []
    for d in found:
        srcs = [picks[i] for i in d["images"]] or picks
        merged.append((d["text"], [{"kind": "quiz", "ref": it["id"], "url": it["image"], "label": it["brand"]} for it in srcs]))
    traits.merge(p, merged, embedder())
    taste.save_profile(p)
    return {"profile": taste.public_profile(p), "read": [d["text"] for d in found]}


# ---------- traits (the personal lens, explained) ----------

class TraitIn(BaseModel):
    text: str


@app.post("/api/traits")
def add_trait(body: TraitIn):
    cleaned = traits.clean([body.text])
    if not cleaned:
        raise HTTPException(400, "Describe the piece itself - cut, fabric, detail, colour - not a style label.")
    p = taste.load_profile()
    if any(t["text"] == cleaned[0] for t in p["traits"]):
        raise HTTPException(409, "You already have that trait.")
    t = traits.new_trait(cleaned[0], embedder(), [{"kind": "user", "ref": "you", "label": "added by you"}], user_added=True)
    p["traits"].append(t)
    taste.save_profile(p)
    return traits.public(t)


class TraitPatch(BaseModel):
    enabled: bool | None = None
    text: str | None = None


@app.patch("/api/traits/{trait_id}")
def patch_trait(trait_id: str, body: TraitPatch):
    p = taste.load_profile()
    t = next((t for t in p["traits"] if t["id"] == trait_id), None)
    if t is None:
        raise HTTPException(404, "unknown trait")
    if body.enabled is not None:
        t["enabled"] = body.enabled
    if body.text is not None:
        cleaned = traits.clean([body.text])
        if not cleaned:
            raise HTTPException(400, "Describe the piece itself - cut, fabric, detail, colour - not a style label.")
        t["text"] = cleaned[0]
        t["embedding"] = embedder().texts([cleaned[0]])[0].tolist()
    taste.save_profile(p)
    return traits.public(t)


@app.delete("/api/traits/{trait_id}")
def delete_trait(trait_id: str):
    p = taste.load_profile()
    p["traits"] = [t for t in p["traits"] if t["id"] != trait_id]
    taste.save_profile(p)
    return {"ok": True}


@app.post("/api/traits/refine")
def refine_traits():
    """Opt-in Claude pass (text only, ~300 tokens): rewrite the free read of screenshots, pins and quiz picks."""
    _require_llm()
    p = taste.load_profile()
    kinds = ("image", "quiz", "pin")
    cands = [t for t in p["traits"] if any(s["kind"] in kinds for s in t["sources"])]
    if not cands:
        raise HTTPException(400, "Nothing to sharpen yet - add a screenshot or take the quiz first.")
    tv = taste_vector(p)
    titles = [catalog.items[i]["title"] for i in np.argsort(-(catalog.E @ tv))[:12]] if tv is not None else []
    found = traits.polish([t["text"] for t in cands], titles)
    if not found:
        raise HTTPException(502, "Claude didn't return anything usable.")
    sources, seen = [], set()
    for t in cands:
        for s in t["sources"]:
            if s["kind"] in kinds and (s["kind"], s["ref"]) not in seen:
                seen.add((s["kind"], s["ref"]))
                sources.append(s)
        t["sources"] = [s for s in t["sources"] if s["kind"] not in kinds]
    p["traits"] = [t for t in p["traits"] if t["sources"] or t.get("user_added")]
    traits.merge(p, [(text, sources) for text in found], embedder())
    taste.save_profile(p)
    return {"profile": taste.public_profile(p), "read": found}


# ---------- brands for you (trait coverage) ----------

class BrandsIn(BaseModel):
    brands: list[str]


@app.post("/api/profile/brands")
def set_brands(body: BrandsIn):
    p = taste.load_profile()
    wanted: dict[str, str] = {}
    for b in body.brands:
        b = b.strip()
        if b and b.lower() not in wanted:
            wanted[b.lower()] = b
    before = {b.lower(): b for b in p["liked_brands"]}
    added = [wanted[k] for k in wanted if k not in before]
    removed = [before[k] for k in before if k not in wanted]
    read: dict[str, list[str]] = {}
    for name in added:
        key, found = traits.from_brand(catalog, name)        # precomputed for every brand we know
        label = key or name
        traits.merge(p, [(t, [{"kind": "brand", "ref": label, "label": label}]) for t in found], embedder())
        read[name] = found
    for name in removed:
        traits.remove_source(p, "brand", name)
    p["liked_brands"] = [catalog.brand_key(b) or b for b in wanted.values()]
    taste.save_profile(p)
    return {"profile": taste.public_profile(p), "read": read}


@app.get("/api/recommend")
def recommend_brands(limit: int = 12):
    p = taste.load_profile()
    exclude = set(p["liked_brands"]) | set(p["dismissed_brands"])
    result = recommend.score_brands(catalog, p["traits"], exclude=exclude, limit=max(1, min(limit, 50)))
    saved = {b.lower() for b in p["saved_brands"]}
    for c in result["brands"]:
        c["saved"] = c["brand"].lower() in saved
    return result


class BrandFeedback(BaseModel):
    brand: str
    action: Literal["save", "dismiss", "clear"]


@app.post("/api/brands/feedback")
def brand_feedback(body: BrandFeedback):
    if body.brand not in catalog.brand_rows:
        raise HTTPException(404, "unknown brand")
    p = taste.load_profile()
    for key in ("saved_brands", "dismissed_brands"):
        p[key] = [b for b in p[key] if b != body.brand]
    if body.action == "save":
        p["saved_brands"].append(body.brand)
        _discover(p, body.brand, "saved the brand")
    elif body.action == "dismiss":
        p["dismissed_brands"].append(body.brand)
    taste.save_profile(p)
    return {"saved": p["saved_brands"], "dismissed": p["dismissed_brands"]}


@app.get("/api/brands/{brand}/items")
def brand_pieces(brand: str, limit: int = 24):
    if brand not in catalog.brand_rows:
        raise HTTPException(404, "unknown brand")
    eng = engine()
    rows = [i for i in catalog.brand_rows[brand] if catalog.items[i]["available"]][:max(1, min(limit, 60))]
    return {"items": [eng.result(i, 0.0, 0.0, {"tokens": []}) for i in rows]}
