"""Brand quality signals from the data we already hold - a drop-shipper triage, not a verdict.

Prints a table (most suspicious first) and writes data/brand_signals.json for the research pass.
Run: python -m app.audit_brands
"""
import json
import re
import statistics
from collections import Counter, defaultdict

from .config import DATA
from .index import Catalog
from .search import to_usd

SIGNALS_FILE = DATA / "brand_signals.json"

SEO_WORDS = re.compile(r"\b(women'?s|womens|ladies|casual|loose|solid color|solid colour|new arrival|fashion|"
                       r"elegant|sexy|plus size|for women|hot sale|streetwear|y2k|vintage style|korean style|"
                       r"harajuku|kawaii|summer|autumn|spring|winter)\b", re.I)
PLACEHOLDER = re.compile(r"[A-Z0-9_]{6,}")


def signals(catalog: Catalog) -> list[dict]:
    rows = []
    for brand, idx in catalog.brand_rows.items():
        items = [catalog.items[i] for i in idx]
        prices = [to_usd(it["price"], it["currency"]) for it in items if it["price"]]
        prices = [p for p in prices if p and p >= 3]
        titles = [it["title"] for it in items]
        words = [len(t.split()) for t in titles]
        vendors = Counter((it.get("vendor") or brand).strip().lower() for it in items)
        own_vendor = vendors.get(brand.lower(), 0) / max(1, len(items))
        meta = catalog.brand_meta.get(brand, {})
        row = {
            "brand": brand,
            "domain": meta.get("domain"),
            "currency": meta.get("currency"),
            "items": len(items),
            "median_usd": round(statistics.median(prices), 0) if prices else None,
            "p10_usd": round(sorted(prices)[len(prices) // 10], 0) if prices else None,
            "seo_titles": round(sum(1 for t in titles if SEO_WORDS.search(t)) / len(titles), 2),
            "long_titles": round(sum(1 for w in words if w >= 8) / len(titles), 2),
            "caps_titles": round(sum(1 for t in titles if t.isupper() or PLACEHOLDER.search(t)) / len(titles), 2),
            "avg_tags": round(sum(len(it.get("tags") or []) for it in items) / len(items), 1),
            "distinct_vendors": len(vendors),
            "own_vendor_share": round(own_vendor, 2),
            "sized": round(sum(1 for it in items if it.get("sizes_offered")) / len(items), 2),
            "tags": meta.get("tags", []),
        }
        # suspicion score: cheap, SEO-stuffed, many vendors, tag spam
        s = 0.0
        if row["median_usd"] is not None:
            s += 3 if row["median_usd"] < 25 else 2 if row["median_usd"] < 40 else 1 if row["median_usd"] < 60 else 0
        s += 3 * row["seo_titles"] + 2 * row["long_titles"]
        s += 2 if row["distinct_vendors"] >= 5 else 1 if row["distinct_vendors"] >= 3 else 0
        s += 1 if row["avg_tags"] >= 15 else 0
        s += 1 if row["caps_titles"] >= 0.5 else 0
        row["suspicion"] = round(s, 2)
        rows.append(row)
    rows.sort(key=lambda r: -r["suspicion"])
    return rows


def main() -> None:
    catalog = Catalog()
    rows = signals(catalog)
    with open(SIGNALS_FILE, "w") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    print(f"{'brand':<24} {'susp':>4} {'med$':>5} {'items':>5} {'seo':>4} {'long':>4} {'vend':>4} {'tags':>4}  sample title")
    for r in rows[:60]:
        sample = next((catalog.items[i]["title"] for i in catalog.brand_rows[r["brand"]]), "")[:48]
        print(f"{r['brand'][:24]:<24} {r['suspicion']:>4} {str(r['median_usd']):>5} {r['items']:>5} {r['seo_titles']:>4} "
              f"{r['long_titles']:>4} {r['distinct_vendors']:>4} {r['avg_tags']:>4}  {sample}")
    print(f"\n{len(rows)} brands -> {SIGNALS_FILE}")


if __name__ == "__main__":
    main()
