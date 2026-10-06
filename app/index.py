"""In-memory product index: metadata + CLIP embeddings, grouped by brand."""
import json
from collections import defaultdict

import numpy as np
import yaml

from . import sizes
from .config import BRANDS_FILE, DATA, EMBEDDINGS_FILE, INDEX_FILE

BRAND_TIERS_FILE = DATA / "brand_tiers.json"     # research pass: key -> {tier, reason, based_in, evidence}
BRAND_REVIEW_FILE = DATA / "brand_review.json"   # owner's decisions on /curate: key -> {decision, ts}


def _load_json(path) -> dict:
    return json.load(open(path)) if path.exists() else {}


def _apply_sizes(it: dict) -> None:
    rows = it.get("sizes") or []
    if not rows:
        it["sizes_in_stock"], it["sizes_offered"] = [], []
        return
    canon = sizes.normalise([r["label"] for r in rows], it.get("currency", "USD"), it.get("title", ""))
    order = lambda s: sizes.LETTERS.index(s) if s in sizes.LETTERS else 99
    for r in rows:
        r["canon"] = canon.get(r["label"], [])
    it["sizes_in_stock"] = sorted({c for r in rows if r["available"] for c in r["canon"]}, key=order)
    it["sizes_offered"] = sorted({c for r in rows for c in r["canon"]}, key=order)


class Catalog:
    def __init__(self, curated: bool = True):
        if not INDEX_FILE.exists() or not EMBEDDINGS_FILE.exists():
            raise FileNotFoundError("No index yet - run `python -m app.crawl` then `python -m app.build`.")
        self.items: list[dict] = [json.loads(line) for line in open(INDEX_FILE)]
        self.E: np.ndarray = np.load(EMBEDDINGS_FILE).astype(np.float32)   # stored fp16 to halve the download
        assert len(self.items) == len(self.E), "index.jsonl and embeddings.npy are out of sync - rerun app.build"
        # Curation: brands graded C by the research pass, or dropped by the owner on /curate, leave the index.
        self.tiers = _load_json(BRAND_TIERS_FILE)
        self.review = _load_json(BRAND_REVIEW_FILE)
        keep = np.array([self.allowed(it["brand"]) if curated else True for it in self.items], dtype=bool)
        self.row_mask = keep                        # over the original index rows; categories.npz follows it
        self.excluded_brands = sorted({it["brand"] for it, k in zip(self.items, keep) if not k})
        self.items = [it for it, k in zip(self.items, keep) if k]
        self.E = self.E[keep]
        from . import categorize                    # (categorize imports Catalog only inside its main)
        self.cats = categorize.load(self.row_mask)
        for it in self.items:                       # sizes are normalised at load, so rules can change without a recrawl
            _apply_sizes(it)
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
