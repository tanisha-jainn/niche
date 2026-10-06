"""Merge researched brand candidates into data/brands.yaml, keeping only stores whose
public /products.json actually answers.

Run: python -m app.grow candidates1.jsonl [candidates2.jsonl ...]
Each input line: {"name", "domain", "country"?, "currency", "tags", "source"?}
"""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import httpx
import yaml

from .config import BRANDS_FILE, DATA
from .crawl import UA

REJECTS_FILE = DATA / "grow_rejects.json"


def normalize_domain(d: str) -> str:
    d = d.strip().lower()
    d = re.sub(r"^https?://", "", d)
    d = d.split("/")[0]
    return d[4:] if d.startswith("www.") else d


def probe(client: httpx.Client, domain: str) -> tuple[str, int | None, str]:
    """Return (domain, product_count, status). Follows redirects so a bare domain that
    301s to www still passes; the redirect target's host is recorded when it differs."""
    try:
        r = client.get(f"https://{domain}/products.json?limit=5")
        if r.status_code != 200:
            return domain, None, str(r.status_code)
        products = r.json().get("products")
        if not isinstance(products, list):
            return domain, None, "no-products-key"
        final = normalize_domain(str(r.url.host))
        return final, len(products), "ok"
    except (httpx.HTTPError, ValueError) as e:
        return domain, None, f"error:{e.__class__.__name__}"


def main(paths: list[str]) -> None:
    with open(BRANDS_FILE) as f:
        existing: list[dict] = yaml.safe_load(f)
    have = {normalize_domain(b["domain"]) for b in existing}
    have_names = {b["name"].lower() for b in existing}

    candidates: dict[str, dict] = {}
    for path in paths:
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except ValueError:
                continue
            d = normalize_domain(c.get("domain", ""))
            if not d or "." not in d or d in have or d in candidates or c["name"].lower() in have_names:
                continue
            candidates[d] = {
                "name": c["name"].strip(),
                "domain": d,
                "currency": (c.get("currency") or "USD").upper(),
                "tags": [t.lower() for t in (c.get("tags") or [])][:5],
                "source": c.get("source", ""),
            }
    print(f"{len(candidates)} new candidate domains after dedupe")

    accepted, rejected = [], []
    with httpx.Client(headers={"User-Agent": UA, "Accept": "application/json"},
                      follow_redirects=True, timeout=15) as client, ThreadPoolExecutor(max_workers=16) as ex:
        for c, (final, n, status) in zip(candidates.values(), ex.map(lambda d: probe(client, d), candidates)):
            if n:
                if final != c["domain"] and final not in have:
                    c["domain"] = final
                if c["domain"] in have:
                    continue
                have.add(c["domain"])
                accepted.append({k: v for k, v in c.items() if k != "source"})
            else:
                rejected.append({**c, "status": status if n is None else "empty"})

    with open(BRANDS_FILE, "w") as f:
        f.write("# Seed list of independent / non-mainstream brands with public Shopify storefronts.\n"
                "# Grown by `python -m app.grow`; stores that fail to answer are listed in data/grow_rejects.json.\n")
        for b in existing + accepted:
            f.write("- " + json.dumps(b, ensure_ascii=False) + "\n")   # JSON is valid YAML flow style
    with open(REJECTS_FILE, "w") as f:
        json.dump(sorted(rejected, key=lambda r: r["status"]), f, indent=2, ensure_ascii=False)

    from collections import Counter
    print(f"accepted {len(accepted)}, rejected {len(rejected)} -> {Counter(r['status'] for r in rejected).most_common(6)}")
    print(f"brands.yaml now has {len(existing) + len(accepted)} stores")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
