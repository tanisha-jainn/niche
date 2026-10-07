"""In-memory product index: metadata + CLIP embeddings, grouped by brand."""
import json
import sys
import threading
from collections import defaultdict

import numpy as np
import yaml

from . import sizes
from .config import BRANDS_FILE, DATA, EMBEDDINGS_FILE, INDEX_FILE

BRAND_TIERS_FILE = DATA / "brand_tiers.json"     # research pass: key -> {tier, reason, based_in, evidence}
BRAND_REVIEW_FILE = DATA / "brand_review.json"   # owner's decisions on /curate: key -> {decision, ts}


def _load_json(path) -> dict:
    return json.load(open(path)) if path.exists() else {}


_shared: dict = {}


def _share(v):
    """One shared copy of repeated values (size tuples, tag tuples) across 45k items."""
    return _shared.setdefault(v, v)


def _apply_sizes(it: dict) -> None:
    """Sizes are normalised at load, so rules can change without a recrawl. Stored compactly as
    ((label, available), ...); size_rows() expands them for the item drawer."""
    rows = it.get("sizes") or []
    it["sizes"] = _share(tuple((sys.intern(r["label"]), bool(r["available"])) for r in rows))
    if not rows:
        it["sizes_in_stock"], it["sizes_offered"] = (), ()
        return
    canon = sizes.normalise([r["label"] for r in rows], it.get("currency", "USD"), it.get("title", ""))
    order = lambda s: sizes.LETTERS.index(s) if s in sizes.LETTERS else 99
    it["sizes_in_stock"] = _share(tuple(sorted({c for r in rows if r["available"] for c in canon.get(r["label"], [])}, key=order)))
    it["sizes_offered"] = _share(tuple(sorted({c for r in rows for c in canon.get(r["label"], [])}, key=order)))


def size_rows(it: dict) -> list[dict]:
    canon = sizes.normalise([lab for lab, _ in it["sizes"]], it.get("currency", "USD"), it.get("title", ""))
    return [{"label": lab, "canon": canon.get(lab, []), "available": avail} for lab, avail in it["sizes"]]


LEAN_KEYS = ("id", "brand", "domain", "title", "url", "image", "price", "currency", "product_type",
             "available", "sizes_in_stock", "sizes_offered")


def _full(it: dict) -> dict:
    """The complete record as the UI needs it (galleries, description, tags, per-size stock)."""
    for k in ("brand", "domain", "currency", "product_type"):
        it[k] = sys.intern(it.get(k) or "")
    it["tags"] = _share(tuple(sys.intern(t) for t in (it.get("tags") or [])[:6]))
    it["images"] = tuple((it.get("images") or [it["image"]])[:5])
    it["image"] = it["images"][0]
    it["description"] = (it.get("description") or "")[:400]
    it.pop("vendor", None)
    it.pop("published_at", None)
    _apply_sizes(it)
    return it


def _lean(it: dict) -> dict:
    """What every item keeps in memory - ranking and filtering only. ~5x smaller than the full record."""
    _full(it)
    return {k: it[k] for k in LEAN_KEYS}


class HalfMatrix:
    """The item-embedding table kept in fp16 (half the memory); `E @ v` and `E[rows]` return fp32,
    computed in small chunks so no full-size fp32 copy ever exists. Rankings are bit-identical."""
    CHUNK = 8192

    def __init__(self, data: np.ndarray):
        self.data = data
        self.shape = data.shape

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> np.ndarray:
        return np.asarray(self.data[idx], dtype=np.float32)

    def __matmul__(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        out = np.empty((len(self.data),) + v.shape[1:], dtype=np.float32)
        for i in range(0, len(self.data), self.CHUNK):
            out[i:i + self.CHUNK] = self.data[i:i + self.CHUNK].astype(np.float32) @ v
        return out


class Catalog:
    def __init__(self, curated: bool = True):
        if not INDEX_FILE.exists() or not EMBEDDINGS_FILE.exists():
            raise FileNotFoundError("No index yet - run `python -m app.crawl` then `python -m app.build`.")
        self.tiers = _load_json(BRAND_TIERS_FILE)
        self.review = _load_json(BRAND_REVIEW_FILE)
        # Stream the index: each record is filtered (curation) and compacted as it's read, so the full
        # 45k raw dicts never sit in memory at once.
        self.items, keep, excluded, offsets = [], [], set(), []
        with open(INDEX_FILE, "rb") as f:
            pos = 0
            for raw in f:
                it = json.loads(raw)
                ok = self.allowed(it["brand"]) if curated else True
                keep.append(ok)
                if ok:
                    self.items.append(_lean(it))
                    offsets.append(pos)
                else:
                    excluded.add(it["brand"])
                pos += len(raw)
        self.offsets = np.array(offsets, dtype=np.int64)
        self._fh = open(INDEX_FILE, "rb")
        self._fh_lock = threading.Lock()
        keep = np.array(keep, dtype=bool)
        E = np.load(EMBEDDINGS_FILE, mmap_mode="r")
        assert len(keep) == len(E), "index.jsonl and embeddings.npy are out of sync - rerun app.build"
        self.E = HalfMatrix(np.ascontiguousarray(E[keep], dtype=np.float16))
        del E
        self.row_mask = keep                        # over the original index rows; categories.npz follows it
        self.excluded_brands = sorted(excluded)
        from . import categorize                    # (categorize imports Catalog only inside its main)
        self.cats = categorize.load(self.row_mask)
        self.by_id = {it["id"]: i for i, it in enumerate(self.items)}
        self.brand_rows: dict[str, list[int]] = defaultdict(list)
        for i, it in enumerate(self.items):
            self.brand_rows[it["brand"]].append(i)
        with open(BRANDS_FILE) as f:
            self.brand_meta = {b["name"]: b for b in yaml.safe_load(f)}

    def brands(self) -> list[dict]:
        return [
            {
                "name": name,
                "domain": self.brand_meta.get(name, {}).get("domain"),
                "tags": self.brand_meta.get(name, {}).get("tags", []),
                "count": len(rows),
            }
            for name, rows in sorted(self.brand_rows.items())
        ]

    def full(self, i: int) -> dict:
        """Complete record for row i, read from disk (only ever needed for what's on screen)."""
        with self._fh_lock:
            self._fh.seek(int(self.offsets[i]))
            raw = self._fh.readline()
        return _full(json.loads(raw))

    def iter_full(self):
        """All kept records in row order, streamed (used once to build the text index)."""
        with open(INDEX_FILE, "rb") as f:
            for off in self.offsets:
                f.seek(int(off))
                yield _full(json.loads(f.readline()))

    def allowed(self, brand: str) -> bool:
        """Owner's decision wins; otherwise the research tier; otherwise in."""
        key = brand.lower()
        decision = (self.review.get(key) or {}).get("decision")
        if decision:
            return decision == "keep"
        t = self.tiers.get(key) or {}
        return not (t.get("tier") == "C" and t.get("verified"))   # an unverified C never removes a brand

    def brand_key(self, name: str) -> str | None:
        """Case-insensitive lookup of an indexed brand's canonical name."""
        lowered = name.strip().lower()
        return next((b for b in self.brand_rows if b.lower() == lowered), None)

    def brand_centroid(self, name: str) -> np.ndarray:
        v = self.E[self.brand_rows[name]].mean(axis=0)
        return v / np.linalg.norm(v)

    def item(self, product_id: str) -> dict | None:
        i = self.by_id.get(product_id)
        return self.items[i] if i is not None else None
