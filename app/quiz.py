"""Taste quiz: a spread of catalog pieces from different corners of the index, for tap-what-you'd-wear onboarding."""
import re

import numpy as np

from .index import Catalog

POOL = 4000
PLACEHOLDER = re.compile(r"[A-Z0-9_]{6,}")   # stores that publish SKU-like titles make bad quiz cards


def spread(catalog: Catalog, n: int = 20, seed: int | None = None) -> list[dict]:
    """k-means over a random in-stock sample, then the most typical piece of each cluster, one brand each."""
    rng = np.random.default_rng(seed)
    candidates = [i for i, it in enumerate(catalog.items)
                  if it["available"] and it["price"] and len(it["title"]) > 8 and not PLACEHOLDER.search(it["title"])]
    pool = rng.choice(candidates, size=min(POOL, len(candidates)), replace=False)
    E = catalog.E[pool]
    centers = E[rng.choice(len(pool), size=n, replace=False)].copy()
    for _ in range(15):
        assign = (E @ centers.T).argmax(axis=1)
        for k in range(n):
            members = E[assign == k]
            if len(members):
                c = members.mean(axis=0)
                centers[k] = c / (np.linalg.norm(c) or 1.0)
    picks, brands = [], set()
    for k in range(n):
        members = np.flatnonzero(assign == k)
        for j in members[np.argsort(-(E[members] @ centers[k]))]:
            it = catalog.items[pool[j]]
            if it["brand"] not in brands:
                brands.add(it["brand"])
                picks.append({k2: it[k2] for k2 in ("id", "title", "image", "brand", "price", "currency")})
                break
    rng.shuffle(picks)
    return picks
