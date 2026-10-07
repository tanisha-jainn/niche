"""Brand recommendations: rank brands by how many of the user's traits they carry, with evidence.

Each enabled trait is a text query over the catalog. A brand "carries" a trait when its best few pieces
score close to the best matches in the whole catalog for that trait - the bar is set per trait, by the
trait's own top results, so a vague trait and a precise one are judged on the same footing. Coverage
(how many traits, weighted) dominates the score; strength breaks ties. The justification is built from
the same structure, so it's always true.
"""
import numpy as np

from . import categorize
from .index import Catalog

FLOOR = 2.5      # absolute z-score floor for a brand's top pieces
RELATIVE = 0.85  # ...and they must reach this fraction of the trait's best catalog-wide match
# Tuned 2026-10-04 on an 18-trait profile: 0.80 -> 242 candidate brands with weak evidence,
# 0.85 -> 158 with mostly-right evidence, 0.90 -> 73 but nothing covered more than 2 traits.
TOP_K = 3        # how many of the brand's best pieces are averaged (robust to one lucky match)
REF_K = 10       # how many catalog-wide top matches define "what a real match looks like"
MIN_ITEMS = 3
WHY_MAX = 3      # traits named in the sentence; the rest are "+N more"


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 1e-6 else 1.0)


def _item(it: dict) -> dict:
    return {k: it[k] for k in ("id", "title", "url", "image", "price", "currency", "product_type")}


def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _category_mask(cats: dict | None, trait_text: str) -> np.ndarray | None:
    """Items that may NOT prove this trait: wrong garment category (when confidently assigned), and
    bags/shoes/accessories for any clothing trait."""
    if cats is None:
        return None
    names, assign, margin = cats["names"], cats["assign"], cats["margin"]
    cat = categorize.trait_category(trait_text)
    confident = margin > np.percentile(margin, 25)
    if cat in categorize.NON_CLOTHING:
        allowed = {cat}
    else:
        non_clothing = np.isin(assign, [names.index(n) for n in categorize.NON_CLOTHING])
        if cat is None:                       # a print / fabric / fit trait: any garment, no accessories
            return non_clothing & confident
        allowed = categorize.COMPATIBLE[cat]
    wrong = ~np.isin(assign, [names.index(n) for n in allowed])
    return wrong & confident


def trait_scores(catalog: Catalog, traits: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Z-scored similarity of every item to every trait (category-masked), plus each trait's bar."""
    V = np.stack([np.asarray(t["embedding"], dtype=np.float32) for t in traits])
    Z = np.stack([_zscore(catalog.E @ v) for v in V])                      # (traits, items)
    for ti, t in enumerate(traits):
        mask = _category_mask(catalog.cats, t["text"])
        if mask is not None:
            Z[ti, mask] = -10.0
    ref = np.sort(Z, axis=1)[:, -REF_K:].mean(axis=1)                       # (traits,)
    bar = np.maximum(FLOOR, RELATIVE * ref)
    return Z, bar


def score_brands(
    catalog: Catalog,
    traits: list[dict],
    exclude: set[str] = frozenset(),
    only: set[str] | None = None,
    limit: int = 12,
) -> dict:
    enabled = [t for t in traits if t.get("enabled", True)]
    if not enabled:
        return {"trait_count": 0, "candidates": 0, "brands": [], "confidence": confidence(enabled)}

    Z, bar = trait_scores(catalog, enabled)
    W = np.array([trait_weight(t) for t in enabled])
    excluded = {b.lower() for b in exclude}
    wanted = {b.lower() for b in only} if only is not None else None

    cards = []
    for brand, rows in catalog.brand_rows.items():
        if wanted is not None and brand.lower() not in wanted:
            continue
        if wanted is None and brand.lower() in excluded:
            continue
        idx = np.array([i for i in rows if catalog.items[i]["available"]] or rows)
        if len(idx) < MIN_ITEMS:
            continue
        Zb = Z[:, idx]                                                      # (traits, brand items)
        strength = np.sort(Zb, axis=1)[:, -min(TOP_K, len(idx)):].mean(axis=1)
        best = idx[Zb.argmax(axis=1)]
        matched = [ti for ti in range(len(enabled)) if strength[ti] >= bar[ti]]
        if not matched and wanted is None:
            continue
        coverage = float(W[matched].sum() / W.sum()) if matched else 0.0
        score = coverage * 10 + 0.5 * float((strength[matched] - bar[matched]).sum()) if matched else 0.0

        prices = [catalog.items[i]["price"] for i in idx if catalog.items[i]["price"]]
        meta = catalog.brand_meta.get(brand, {})
        matched_sorted = sorted(matched, key=lambda ti: -(strength[ti] - bar[ti]))
        cards.append({
            "brand": brand,
            "domain": meta.get("domain"),
            "url": f"https://{meta['domain']}" if meta.get("domain") else None,
            "currency": meta.get("currency", "USD"),
            "tags": meta.get("tags", []),
            "count": int(len(rows)),
            "price_low": int(np.percentile(prices, 15)) if prices else None,
            "price_high": int(np.percentile(prices, 85)) if prices else None,
            "coverage": f"{len(matched)} of {len(enabled)}",
            "score": round(score, 2),
            "matched": [{"trait_id": enabled[ti]["id"], "trait": enabled[ti]["text"],
                         "strength": round(float(strength[ti]), 2), "bar": round(float(bar[ti]), 2),
                         "evidence": _item(catalog.full(int(best[ti])))} for ti in matched_sorted],
            "missing": [enabled[ti]["text"] for ti in range(len(enabled)) if ti not in matched],
        })

    cards.sort(key=lambda c: -c["score"])
    for c in cards:
        names = [m["trait"] for m in c["matched"]]
        if not names:
            c["why"] = "None of your traits seen here yet."
        else:
            extra = f" (+{len(names) - WHY_MAX} more)" if len(names) > WHY_MAX else ""
            c["why"] = f"Carries {_join(names[:WHY_MAX])}{extra} — {c['coverage']} of your traits."
    return {"trait_count": len(enabled), "candidates": len(cards), "brands": cards[:limit],
            "confidence": confidence(enabled)}


def trait_weight(t: dict) -> float:
    """A trait seen in several screenshots or brands counts for more; no manual weighting."""
    return min(3.0, 1.0 + 0.5 * max(0, len(t.get("sources", [])) - 1))


def confidence(enabled: list[dict]) -> dict:
    sources = {(s["kind"], s.get("ref")) for t in enabled for s in t.get("sources", [])}
    n_traits, n_sources = len(enabled), len(sources)
    if n_traits >= 5 and n_sources >= 3:
        return {"level": "high", "note": f"based on {n_traits} traits from {n_sources} sources"}
    if n_traits >= 3:
        return {"level": "medium", "note": f"based on {n_traits} traits from {n_sources} source{'s' if n_sources != 1 else ''} — a couple more screenshots or brands will sharpen this"}
    if n_traits:
        return {"level": "low", "note": f"based on only {n_traits} trait{'s' if n_traits != 1 else ''} so far — add a few screenshots or a brand you shop"}
    return {"level": "none", "note": "nothing to go on yet"}


def brand_items(catalog: Catalog, brand: str, trait: dict | None = None, limit: int = 12) -> list[dict]:
    """Pieces from one brand - ranked by a trait if given, else in-stock catalog order."""
    rows = [i for i in catalog.brand_rows.get(brand, []) if catalog.items[i]["available"]]
    if trait is not None and rows:
        z = catalog.E[rows] @ np.asarray(trait["embedding"], dtype=np.float32)
        rows = [rows[i] for i in np.argsort(-z)]
    return [_item(catalog.full(i)) for i in rows[:limit]]
