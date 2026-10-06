"""Crawl each seed store's public Shopify /products.json into data/catalog.jsonl.

Run: python -m app.crawl
"""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import yaml

from . import sizes
from .config import (
    BRANDS_FILE, CATALOG_FILE, CRAWL_REPORT_FILE, DATA, IMAGE_WIDTH, MAX_PRODUCTS_PER_BRAND,
)

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
TAG_RE = re.compile(r"<[^>]+>")
SKIP_WORDS = ("gift card", "giftcard", "e-gift", "gift voucher")
PAGE_SIZE = 250
MAX_IMAGES = 6


def load_brands() -> list[dict]:
    with open(BRANDS_FILE) as f:
        return yaml.safe_load(f)


def image_url(src: str, width: int = IMAGE_WIDTH) -> str:
    # Shopify's CDN resizes on the fly via a query param; keeps downloads small.
    sep = "&" if "?" in src else "?"
    return f"{src}{sep}width={width}"


def _size_order(s: str) -> int:
    return sizes.LETTERS.index(s) if s in sizes.LETTERS else 99


def _variants(p: dict, currency: str, title: str) -> tuple[list[dict], list[str], list[str]]:
    """Per-size stock from Shopify variants: [{label, canon, available}], sizes in stock, sizes offered."""
    variants = p.get("variants") or []
    pos = sizes.size_option_index(p.get("options") or [])
    by_label: dict[str, bool] = {}
    for v in variants:
        label = str(v.get(f"option{pos}") or "").strip() if pos else ""
        if not label and len(variants) == 1:
            label = str(v.get("title") or "").strip()
        if not label:
            continue
        by_label[label] = by_label.get(label, False) or bool(v.get("available"))
    canon = sizes.normalise(list(by_label), currency, title)
    rows = [{"label": lab, "canon": canon.get(lab, []), "available": avail} for lab, avail in by_label.items()]
    in_stock = sorted({c for r in rows if r["available"] for c in r["canon"]}, key=_size_order)
    offered = sorted({c for r in rows for c in r["canon"]}, key=_size_order)
    return rows, in_stock, offered


def normalize(p: dict, brand: dict) -> dict | None:
    if not p.get("images"):
        return None
    title = p.get("title") or ""
    ptype = p.get("product_type") or ""
    haystack = f"{title} {ptype}".lower()
    if any(w in haystack for w in SKIP_WORDS):
        return None
    variants = p.get("variants") or []
    price = None
    for v in variants:
        try:
            price = float(v["price"])
            break
        except (KeyError, TypeError, ValueError):
            continue
    available = any(v.get("available") for v in variants) if variants else True
    desc = re.sub(r"\s+", " ", TAG_RE.sub(" ", p.get("body_html") or "")).strip()[:700]
    currency = brand.get("currency", "USD")
    size_rows, in_stock, offered = _variants(p, currency, title)
    return {
        "id": f"{brand['domain']}#{p['id']}",
        "brand": brand["name"],
        "domain": brand["domain"],
        "title": title,
        "url": f"https://{brand['domain']}/products/{p['handle']}",
        "image": image_url(p["images"][0]["src"]),
        "images": [image_url(im["src"]) for im in p["images"][:MAX_IMAGES]],
        "price": price,
        "currency": currency,
        "product_type": ptype,
        "vendor": (p.get("vendor") or "").strip(),   # many distinct vendors on one store = drop-shipping tell
        "tags": p.get("tags") or [],
        "available": available,
        "description": desc,
        "sizes": size_rows,
        "sizes_in_stock": in_stock,
        "sizes_offered": offered,
        "published_at": p.get("published_at"),
    }


def fetch_store(brand: dict, client: httpx.Client) -> tuple[list[dict], dict]:
    domain = brand["domain"]
    products: list[dict] = []
    page, status = 1, None
    while len(products) < MAX_PRODUCTS_PER_BRAND:
        url = f"https://{domain}/products.json?limit={PAGE_SIZE}&page={page}"
        try:
            r = client.get(url)
            status = r.status_code
            if r.status_code != 200:
                break
            batch = r.json().get("products", [])
        except (httpx.HTTPError, ValueError) as e:
            status = f"error:{e.__class__.__name__}"
            break
        if not batch:
            break
        products.extend(item for p in batch if (item := normalize(p, brand)))
        if len(batch) < PAGE_SIZE:
            break
        page += 1
        time.sleep(0.5)
    products = products[:MAX_PRODUCTS_PER_BRAND]
    return products, {"domain": domain, "brand": brand["name"], "status": status, "products": len(products)}


def main() -> None:
    brands = load_brands()
    DATA.mkdir(exist_ok=True)
    all_products: list[dict] = []
    report: list[dict] = []
    headers = {"User-Agent": UA, "Accept": "application/json"}
    with httpx.Client(headers=headers, follow_redirects=True, timeout=20) as client, \
            ThreadPoolExecutor(max_workers=6) as ex:
        futures = [ex.submit(fetch_store, b, client) for b in brands]
        for fut in as_completed(futures):
            products, info = fut.result()
            report.append(info)
            all_products.extend(products)
            print(f"{info['domain']:<34} {str(info['status']):<10} {info['products']:>4}", flush=True)

    with open(CATALOG_FILE, "w") as f:
        for p in all_products:
            f.write(json.dumps(p) + "\n")
    with open(CRAWL_REPORT_FILE, "w") as f:
        json.dump(sorted(report, key=lambda r: r["domain"]), f, indent=2)
    ok = sum(1 for r in report if r["products"])
    sized = sum(1 for p in all_products if p["sizes_offered"])
    print(f"\n{len(all_products)} products from {ok}/{len(brands)} stores ({sized} with recognisable sizes) -> {CATALOG_FILE}")


if __name__ == "__main__":
    main()
