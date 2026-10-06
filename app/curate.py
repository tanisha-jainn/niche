"""Brand curation: the research tiers + the owner's keep/drop decisions, with previews for /curate.
Reads the FULL index (not the curated Catalog) so dropped brands can be reviewed and restored."""
import json
import statistics
import time
from collections import Counter, defaultdict

from .config import INDEX_FILE
from .index import BRAND_REVIEW_FILE, BRAND_TIERS_FILE, _load_json
from .search import to_usd
from . import traits as traits_mod

_overview: list[dict] | None = None


def overview() -> list[dict]:
    """One row per brand in the full index: previews, price, items, tier, decision."""
    global _overview
    if _overview is None:
        by: dict[str, list[dict]] = defaultdict(list)
        for line in open(INDEX_FILE):
            it = json.loads(line)
            by[it["brand"]].append(it)
        rows = []
        for brand, items in sorted(by.items()):
            prices = [p for it in items if (p := to_usd(it["price"], it["currency"])) and p >= 3]
            avail = [it for it in items if it["available"]] or items
            vendors = Counter((it.get("vendor") or "").strip() for it in items if it.get("vendor"))
            others = [v for v, _ in vendors.most_common() if v.lower() != brand.lower()]
            rows.append({
                "brand": brand, "key": brand.lower(), "domain": items[0]["domain"], "currency": items[0]["currency"],
                "items": len(items), "median_usd": round(statistics.median(prices)) if prices else None,
                "vendors": len(vendors), "other_vendors": others[:5],
                "preview": [it["image"] for it in avail[:6]],
                "sample_titles": [it["title"][:60] for it in avail[:4]],
                "traits": traits_mod.cache_get(traits_mod.BRAND_TRAITS_CACHE, brand.lower()) or [],
            })
        _overview = rows
    tiers, review = _load_json(BRAND_TIERS_FILE), _load_json(BRAND_REVIEW_FILE)
    out = []
    for r in _overview:
        t, d = tiers.get(r["key"], {}), review.get(r["key"], {})
        out.append({**r, "tier": t.get("tier"), "reason": t.get("reason", ""), "based_in": t.get("based_in", ""),
                    "evidence": t.get("evidence", ""), "verified": bool(t.get("verified")), "decision": d.get("decision")})
    return out


def decide(key: str, decision: str | None) -> dict:
    review = _load_json(BRAND_REVIEW_FILE)
    if decision in ("keep", "drop"):
        review[key] = {"decision": decision, "ts": time.time()}
    else:
        review.pop(key, None)
    with open(BRAND_REVIEW_FILE, "w") as f:
        json.dump(review, f, indent=1, ensure_ascii=False)
    return review


def summary() -> dict:
    rows = overview()
    tiers = defaultdict(int)
    for r in rows:
        tiers[r["tier"] or "?"] += 1
    decided = sum(1 for r in rows if r["decision"])
    live = sum(1 for r in rows if (r["decision"] or ("drop" if r["tier"] == "C" and r["verified"] else "keep")) == "keep")
    return {"brands": len(rows), "tiers": dict(tiers), "decided": decided, "live": live,
            "verified": sum(1 for r in rows if r["verified"])}
