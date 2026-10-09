"""Relevance-first product search over the whole index. No paid placement, ever.

Ranking = visual meaning (CLIP text->image, so "fall jacket" finds shackets it has never seen the word for)
+ literal text match on titles/types/tags (so "low-rise" is taken at its word)
+ a category gate ("pants" never returns a bodysuit)
+ an optional, small personal nudge from the user's saved looks - applied after relevance, never before
+ a per-brand cap, so a results page is twenty brands, not one.
"""
import math
import re
import sys
from array import array
from collections import Counter, defaultdict

import numpy as np

from . import categorize
from .embed import Embedder
from .index import Catalog

TEXT_WEIGHT = 2.5       # a perfect text match is worth ~2.5 z of visual similarity
PHRASE_BONUS = 0.8      # query bigram appears verbatim in the title
COLOUR_BONUS = 0.5      # queried colour appears in the title
TASTE_WEIGHT = 0.6      # personalisation nudge (z of similarity to the user's saved looks)
PER_BRAND_PER_PAGE = 2
PAGE = 24

COLOURS = {"black", "white", "cream", "ivory", "ecru", "beige", "tan", "camel", "brown", "chocolate", "navy", "blue",
           "red", "burgundy", "wine", "pink", "blush", "rose", "green", "olive", "sage", "khaki", "grey", "gray",
           "charcoal", "silver", "gold", "yellow", "orange", "rust", "purple", "lilac", "lavender", "mint", "oat"}
STOP = {"a", "an", "the", "and", "or", "for", "with", "in", "of", "to", "my", "me", "some", "that", "this", "i", "want", "looking"}
TOKEN = re.compile(r"[a-z0-9]+")
RELEVANT_Z = 1.5        # without a text hit, an item needs this much visual similarity to count as a result
MIN_USD = 5.0           # below this it's a fee, a swatch or a placeholder, not a garment
JUNK = re.compile(r"shipping protection|route package|customi[sz]ation|gift card|donation|sample sale ticket|"
                  r"deposit|e-?gift|\b[A-Z0-9_]{6,}\b")

# To USD, approximate. Display only - the brand's own price is always shown too.
FX = {"USD": 1.0, "CAD": 0.73, "GBP": 1.27, "EUR": 1.08, "AUD": 0.65, "NZD": 0.60, "JPY": 0.0067, "INR": 0.012,
      "KRW": 0.00073, "SEK": 0.095, "DKK": 0.145, "NOK": 0.093, "CHF": 1.12, "PLN": 0.25, "BRL": 0.18, "MXN": 0.055,
      "SGD": 0.75, "HKD": 0.128, "CNY": 0.14, "THB": 0.028, "PHP": 0.017, "IDR": 0.000063, "VND": 0.00004,
      "MYR": 0.21, "TRY": 0.03, "ZAR": 0.055, "COP": 0.00024, "PEN": 0.27, "ARS": 0.001}


def to_usd(price: float | None, currency: str) -> float | None:
    return None if price is None else price * FX.get(currency, 1.0)


def stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokens(text: str) -> list[str]:
    return [stem(t) for t in TOKEN.findall(text.lower()) if t not in STOP]


class TextIndex:
    """Small BM25 over title (x3), product type (x2), tags (x1), description (x0.5)."""

    def __init__(self, catalog: Catalog):
        # Build postings in flat typed arrays (no per-entry Python tuples), then group by token with numpy.
        vocab: dict[str, int] = {}
        tok_ids, docs, weights = array("i"), array("i"), array("f")
        self.length = np.zeros(len(catalog.items), dtype=np.float32)
        self.title_tokens: list[tuple[str, ...]] = []
        for i, it in enumerate(catalog.iter_full()):
            tf: Counter = Counter()
            title = tuple(sys.intern(t) for t in tokens(it["title"]))
            self.title_tokens.append(title)
            for t in title:
                tf[t] += 3.0
            for t in tokens(it.get("product_type") or ""):
                tf[t] += 2.0
            for t in tokens(" ".join(it.get("tags") or [])):
                tf[t] += 1.0
            for t in tokens((it.get("description") or "")[:300]):
                tf[t] += 0.5
            self.length[i] = sum(tf.values())
            for t, w in tf.items():
                tok_ids.append(vocab.setdefault(t, len(vocab)))
                docs.append(i)
                weights.append(w)
        tid = np.frombuffer(tok_ids, dtype=np.int32)
        order = np.argsort(tid, kind="stable")
        tid, d, w = tid[order], np.frombuffer(docs, dtype=np.int32)[order], np.frombuffer(weights, dtype=np.float32)[order]
        bounds = np.searchsorted(tid, np.arange(len(vocab) + 1))
        self.postings = {t: (d[bounds[k]:bounds[k + 1]], w[bounds[k]:bounds[k + 1]]) for t, k in vocab.items()}
        self.avg_len = float(self.length.mean()) if len(self.length) else 1.0
        self.N = len(catalog.items)

    def score(self, query_tokens: list[str]) -> np.ndarray:
        s = np.zeros(self.N, dtype=np.float32)
        k1, b = 1.2, 0.75
        for t in set(query_tokens):
            hit = self.postings.get(t)
            if hit is None:
                continue
            idx, tf = hit
            idf = math.log(1 + (self.N - len(idx) + 0.5) / (len(idx) + 0.5))
            s[idx] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * self.length[idx] / self.avg_len))
        return s


class Search:
    def __init__(self, catalog: Catalog, embedder: Embedder):
        self.catalog = catalog
        self.embedder = embedder
        self.text = TextIndex(catalog)
        self.cats = catalog.cats
        self.usd = np.array([to_usd(it["price"], it["currency"]) or 0.0 for it in catalog.items], dtype=np.float32)
        self.available = np.array([bool(it["available"]) and not JUNK.search(it["title"]) for it in catalog.items])
        self.available &= self.usd >= MIN_USD
        # garment named in the title, else in the store's own product type ("Briefs" -> lingerie)
        self.title_cat = [categorize.trait_category(it["title"]) or categorize.trait_category(it["product_type"])
                          for it in catalog.items]
        self._query_cache: dict[str, np.ndarray] = {}

    # ---------- query understanding ----------

    def parse(self, q: str) -> dict:
        toks = tokens(q)
        return {
            "tokens": toks,
            "bigrams": [f"{a} {b}" for a, b in zip(toks, toks[1:])],
            "colours": [t for t in TOKEN.findall(q.lower()) if t in COLOURS],
            "category": categorize.trait_category(q),
        }

    def category_mask(self, category: str | None) -> np.ndarray | None:
        if not category or self.cats is None:
            return None
        names, assign, margin = self.cats["names"], self.cats["assign"], self.cats["margin"]
        allowed = categorize.COMPATIBLE.get(category, {category})
        ok = np.isin(assign, [names.index(n) for n in allowed])
        unsure = margin <= np.percentile(margin, 25)      # low-confidence categorisation: don't exclude...
        title_says_no = np.array([tc is not None and tc not in allowed for tc in self.title_cat])
        return (ok | unsure) & ~title_says_no              # ...unless the title itself names another garment

    def clip_scores(self, q: str) -> np.ndarray:
        if q not in self._query_cache:
            v = self.embedder.texts([q])[0]
            self._query_cache[q] = self.catalog.E @ v
        return self._query_cache[q]

    # ---------- ranking ----------

    def run(self, q: str, *, category: str | None = None, sizes: dict | None = None, size_filter: bool = False,
            include_unknown_size: bool = True, min_usd: float | None = None, max_usd: float | None = None,
            taste: np.ndarray | None = None, brands: set[str] | None = None, exclude_brands: set[str] | None = None,
            sort: str = "relevance", page: int = 0) -> dict:
        cat = self.catalog
        parsed = self.parse(q)
        category = category or parsed["category"]

        # 1. visual relevance
        sims = self.clip_scores(q)
        z = (sims - sims.mean()) / (sims.std() + 1e-6)
        score = z.copy()
        # 2. literal text relevance
        text = self.text.score(parsed["tokens"])
        if text.max() > 0:
            score += TEXT_WEIGHT * (text / text.max())
        # 3. phrase + colour bonuses on the title
        if parsed["bigrams"] or parsed["colours"]:
            for i, tt in enumerate(self.text.title_tokens):
                if text[i] <= 0:
                    continue
                joined = " ".join(tt)
                score[i] += PHRASE_BONUS * sum(1 for bg in parsed["bigrams"] if bg in joined)
                score[i] += COLOUR_BONUS * sum(1 for c in parsed["colours"] if stem(c) in tt)
        # 4. personal nudge, after relevance
        if taste is not None:
            ts = cat.E @ taste
            score += TASTE_WEIGHT * (ts - ts.mean()) / (ts.std() + 1e-6)

        # 5. hard filters - and only items that are actually relevant count as results
        keep = self.available & ((text > 0) | (z >= RELEVANT_Z))
        if brands:                                          # browsing one brand: everything it has
            keep = self.available.copy()
        m = self.category_mask(category)
        if m is not None:
            keep &= m
        if min_usd is not None:
            keep &= self.usd >= min_usd
        if max_usd is not None:
            keep &= (self.usd <= max_usd) & (self.usd > 0)
        if brands:
            keep &= np.array([it["brand"] in brands for it in cat.items])
        if exclude_brands:
            keep &= np.array([it["brand"] not in exclude_brands for it in cat.items])
        if size_filter and sizes:
            keep &= self.size_mask(sizes, include_unknown_size)
        score[~keep] = -np.inf

        # 6. order + brand diversity
        if sort == "price_asc":
            order = np.lexsort((-score, np.where(keep, self.usd, np.inf)))
        elif sort == "price_desc":
            order = np.lexsort((-score, np.where(keep, -self.usd, np.inf)))
        else:
            order = np.argsort(-score)
        ranked = [int(i) for i in order if np.isfinite(score[i])]
        diversified = self.diversify(ranked)
        total = len(diversified)
        page_rows = diversified[page * PAGE:(page + 1) * PAGE]
        return {
            "query": q, "category": category, "total": total, "page": page, "pages": math.ceil(total / PAGE),
            "brands_on_page": len({cat.items[i]["brand"] for i in page_rows}),
            "items": [self.result(i, score[i], text[i], parsed) for i in page_rows],
        }

    def diversify(self, ranked: list[int]) -> list[int]:
        """At most PER_BRAND_PER_PAGE pieces per brand per page, in relevance order; one card per
        piece (a second colourway of the same title is dropped)."""
        out, deferred = [], []
        counts: Counter = Counter()
        seen_titles: set[tuple[str, str]] = set()
        page_no = 0
        for i in ranked:
            b = self.catalog.items[i]["brand"]
            key = (b, re.sub(r"\s*[-|~(].*$", "", self.catalog.items[i]["title"].lower()).strip())
            if key in seen_titles:
                continue
            seen_titles.add(key)
            if counts[(page_no, b)] < PER_BRAND_PER_PAGE:
                out.append(i)
                counts[(page_no, b)] += 1
            else:
                deferred.append(i)
            if len(out) % PAGE == 0 and out:
                page_no = len(out) // PAGE
                deferred, pending = [], deferred
                for j in pending:
                    bj = self.catalog.items[j]["brand"]
                    if counts[(page_no, bj)] < PER_BRAND_PER_PAGE:
                        out.append(j)
                        counts[(page_no, bj)] += 1
                    else:
                        deferred.append(j)
        return out + deferred

    def size_mask(self, sizes: dict, include_unknown: bool) -> np.ndarray:
        """sizes = {"tops": "S", "bottoms": "M", "dresses": "S"}; item group from its category."""
        names, assign = (self.cats["names"], self.cats["assign"]) if self.cats else (None, None)
        group_of = {"dress": "dresses", "jumpsuit": "dresses", "set": "dresses", "top": "tops", "knitwear": "tops",
                    "jacket": "tops", "coat": "tops", "blazer": "tops", "swimwear": "tops", "lingerie": "tops",
                    "activewear": "tops", "trousers": "bottoms", "jeans": "bottoms", "skirt": "bottoms", "shorts": "bottoms"}
        mask = np.ones(len(self.catalog.items), dtype=bool)
        for i, it in enumerate(self.catalog.items):
            cname = names[assign[i]] if names is not None else None
            group = group_of.get(cname)
            want = sizes.get(group) if group else None
            if not want:
                continue                                   # no preference for this garment group
            stock = it.get("sizes_in_stock") or []
            if not stock and not it.get("sizes_offered"):
                mask[i] = include_unknown
            else:
                mask[i] = want in stock or "ONE" in stock
        return mask

    def result(self, i: int, score: float, text_score: float, parsed: dict) -> dict:
        it = self.catalog.full(i)
        why = []
        tt = set(self.text.title_tokens[i])
        hits = [t for t in parsed["tokens"] if t in tt]
        if hits:
            why.append("title matches " + ", ".join(dict.fromkeys(hits)))
        elif text_score > 0:
            why.append("tagged as a match")
        else:
            why.append("looks like what you asked for")
        return {
            "id": it["id"], "brand": it["brand"], "title": it["title"], "url": it["url"],
            "image": it["image"], "images": list(it["images"]),
            "price": it["price"], "currency": it["currency"], "price_usd": round(float(self.usd[i]), 2) or None,
            "sizes_in_stock": it.get("sizes_in_stock") or [], "sizes_offered": it.get("sizes_offered") or [],
            "product_type": it.get("product_type") or "", "score": round(float(score), 2), "why": why,
            "category": self.cats["names"][self.cats["assign"][i]] if self.cats else None,
        }

    def similar(self, item_id: str, limit: int = 12, same_brand: bool = False) -> list[dict]:
        cat = self.catalog
        i = cat.by_id.get(item_id)
        if i is None:
            return []
        sims = cat.E @ cat.E[i]
        brand = cat.items[i]["brand"]
        out, seen = [], Counter()
        for j in np.argsort(-sims):
            j = int(j)
            if j == i or not cat.items[j]["available"]:
                continue
            bj = cat.items[j]["brand"]
            if same_brand != (bj == brand):
                continue
            if not same_brand and seen[bj] >= 1:
                continue
            seen[bj] += 1
            out.append(self.result(j, sims[j], 0.0, {"tokens": []}))
            if len(out) >= limit:
                break
        return out
