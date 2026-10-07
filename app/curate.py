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
        # stream the full index (it isn't held in memory) keeping only a few fields per brand
        agg: dict[str, dict] = {}
        for line in open(INDEX_FILE):
            it = json.loads(line)
            a = agg.setdefault(it["brand"], {"domain": it["domain"], "currency": it["currency"], "n": 0,
                                             "prices": [], "vendors": Counter(), "preview": [], "titles": []})
            a["n"] += 1
            if (p := to_usd(it["price"], it["currency"])) and p >= 3:
                a["prices"].append(p)
            if it.get("vendor"):
                a["vendors"][it["vendor"].strip()] += 1
            if it["available"] and len(a["preview"]) < 6:
                a["preview"].append(it["image"])
                a["titles"].append(it["title"][:60])
        rows = []
        for brand, a in sorted(agg.items()):
            others = [v for v, _ in a["vendors"].most_common() if v.lower() != brand.lower()]
            rows.append({
                "brand": brand, "key": brand.lower(), "domain": a["domain"], "currency": a["currency"],
                "items": a["n"], "median_usd": round(statistics.median(a["prices"])) if a["prices"] else None,
                "vendors": len(a["vendors"]), "other_vendors": others[:5],
                "preview": a["preview"], "sample_titles": a["titles"][:4],
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
