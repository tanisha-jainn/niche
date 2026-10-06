"""Coverage audit: for each mainstream brand a user might name, how much of the index looks like it?

Prints, per known brand, the number of items with a strong match (z > threshold) and how many distinct
brands those come from - low numbers mark aesthetics the seed list is thin on.
Run: python -m app.audit
"""
import numpy as np

from . import taste
from .embed import Embedder
from .index import Catalog

Z_STRONG = 2.5


def main() -> None:
    cat = Catalog()
    emb = Embedder()
    known = taste.known_brands()
    names = list(known)
    T = emb.texts([known[n] for n in names])
    print(f"{len(cat.items)} items, {len(cat.brand_rows)} brands, model {emb.name}\n")
    print(f"{'known brand':<22} {'strong':>6} {'brands':>6}  top matches")
    rows = []
    for name, t in zip(names, T):
        z = taste._zscore(cat.E @ t)
        strong = np.flatnonzero(z > Z_STRONG)
        top = np.argsort(-z)[:50]
        rows.append((name, len(strong), len({cat.items[i]["brand"] for i in strong}),
                     ", ".join(dict.fromkeys(cat.items[i]["brand"] for i in top[:12]))))
    for name, n, b, top in sorted(rows, key=lambda r: r[1]):
        print(f"{name:<22} {n:>6} {b:>6}  {top[:80]}")


if __name__ == "__main__":
    main()
