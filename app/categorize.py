"""Garment category for every indexed item, via CLIP zero-shot. Free and local.

Used so a brand can only "prove" a trousers trait with trousers, and so bags/shoes/accessories never
stand in for clothing. Run after app.build: python -m app.categorize
"""
import json

import numpy as np

from .config import DATA

CATEGORIES_FILE = DATA / "categories.npz"

# category -> zero-shot prompt
CATEGORIES = {
    "dress": "a dress",
    "top": "a top, blouse, t-shirt, shirt, tank or bodysuit",
    "knitwear": "a knitted sweater, jumper or cardigan",
    "trousers": "trousers or pants",
    "jeans": "denim jeans",
    "skirt": "a skirt",
    "shorts": "shorts",
    "jacket": "a jacket",
    "coat": "a long coat",
    "blazer": "a blazer or suit jacket",
    "jumpsuit": "a jumpsuit, romper or overalls",
    "set": "a matching two-piece outfit set",
    "swimwear": "a swimsuit or bikini",
    "lingerie": "lingerie, underwear or a nightgown",
    "activewear": "leggings, sports bra or gym clothes",
    "shoes": "shoes, boots or sandals",
    "bag": "a handbag or bag",
    "accessory": "an accessory such as a hat, belt, scarf, sunglasses or jewellery",
}

# which item categories may prove a trait of a given category
COMPATIBLE = {
    "dress": {"dress"},
    "top": {"top", "knitwear", "set"},
    "knitwear": {"knitwear", "top"},
    "trousers": {"trousers", "jeans", "set"},
    "jeans": {"jeans", "trousers"},
    "skirt": {"skirt", "set"},
    "shorts": {"shorts"},
    "jacket": {"jacket", "coat", "blazer"},
    "coat": {"coat", "jacket"},
    "blazer": {"blazer", "jacket"},
    "jumpsuit": {"jumpsuit"},
    "set": {"set"},
    "swimwear": {"swimwear"},
    "lingerie": {"lingerie", "dress"},
    "activewear": {"activewear"},
    "shoes": {"shoes"},
    "bag": {"bag"},
    "accessory": {"accessory"},
}
NON_CLOTHING = {"shoes", "bag", "accessory"}

# trait/title/query text -> category, by keyword (CLIP text-to-text similarity is too weak to trust).
# Garment nouns first so "denim lace blouse" is a top; fabric words (denim, knit, slip) are a fallback.
KEYWORDS = [
    ("jumpsuit", "jumpsuit"), ("romper", "jumpsuit"), ("playsuit", "jumpsuit"), ("overalls", "jumpsuit"), ("dungaree", "jumpsuit"),
    ("swimsuit", "swimwear"), ("bikini", "swimwear"), ("swimwear", "swimwear"), ("one-piece", "swimwear"),
    ("nightgown", "lingerie"), ("lingerie", "lingerie"), ("bralette", "lingerie"), ("underwear", "lingerie"), ("brief", "lingerie"), ("thong", "lingerie"),
    ("leggings", "activewear"), ("sports bra", "activewear"), ("bike shorts", "activewear"), ("activewear", "activewear"),
    ("two-piece", "set"), ("co-ord", "set"), ("matching set", "set"), (" set", "set"),
    ("dress", "dress"), ("gown", "dress"),
    ("skirt", "skirt"), ("skort", "skirt"),
    ("shorts", "shorts"),
    ("jeans", "jeans"), ("jean ", "jeans"),
    ("trousers", "trousers"), ("trouser", "trousers"), ("pants", "trousers"), ("pant ", "trousers"), ("slacks", "trousers"), ("culotte", "trousers"), ("jogger", "trousers"),
    ("blazer", "blazer"),
    ("trench", "coat"), ("overcoat", "coat"), ("coat", "coat"),
    ("jacket", "jacket"), ("bomber", "jacket"), ("puffer", "jacket"), ("anorak", "jacket"), ("parka", "jacket"), ("gilet", "jacket"), ("vest", "jacket"),
    ("cardigan", "knitwear"), ("sweater", "knitwear"), ("jumper", "knitwear"), ("pullover", "knitwear"), ("turtleneck", "knitwear"),
    ("blouse", "top"), ("shirt", "top"), ("t-shirt", "top"), ("tee", "top"), ("tank", "top"), ("camisole", "top"), ("cami", "top"),
    ("bodysuit", "top"), ("corset", "top"), ("bustier", "top"), ("hoodie", "top"), ("sweatshirt", "top"), ("polo", "top"), ("crop top", "top"), ("top", "top"),
    ("sneaker", "shoes"), ("loafer", "shoes"), ("sandal", "shoes"), ("boot", "shoes"), ("heel", "shoes"), ("mule", "shoes"), ("shoe", "shoes"),
    ("tote", "bag"), ("handbag", "bag"), ("bag", "bag"),
    ("hat", "accessory"), ("belt", "accessory"), ("scarf", "accessory"), ("jewellery", "accessory"), ("jewelry", "accessory"), ("earring", "accessory"), ("necklace", "accessory"), ("sunglasses", "accessory"),
    # fabric / construction fallbacks
    ("denim", "jeans"), ("knit", "knitwear"), ("slip", "dress"),
]


def trait_category(text: str) -> str | None:
    t = " " + text.lower() + " "
    for kw, cat in KEYWORDS:
        if kw in t:
            return cat
    return None


def load(row_mask: np.ndarray | None = None) -> dict | None:
    """Per-item categories, aligned to data/index.jsonl rows; pass the Catalog's row mask when brands
    have been filtered out so the arrays line up with catalog.items."""
    if not CATEGORIES_FILE.exists():
        return None
    z = np.load(CATEGORIES_FILE, allow_pickle=False)
    assign, margin = z["assign"], z["margin"]
    if row_mask is not None:
        if len(row_mask) != len(assign):
            return None                      # stale categories.npz - rerun app.categorize
        assign, margin = assign[row_mask], margin[row_mask]
    return {"names": list(z["names"]), "assign": assign, "margin": margin}


def main() -> None:
    from .embed import Embedder
    from .index import Catalog
    catalog = Catalog(curated=False)
    embedder = Embedder()
    names = list(CATEGORIES)
    T = embedder.texts([f"a photo of {CATEGORIES[n]}" for n in names])
    S = catalog.E @ T.T                                   # (items, categories)
    top2 = np.sort(S, axis=1)[:, -2:]
    assign = S.argmax(axis=1).astype(np.int16)
    margin = (top2[:, 1] - top2[:, 0]).astype(np.float32)
    np.savez(CATEGORIES_FILE, names=np.array(names), assign=assign, margin=margin)
    counts = {n: int((assign == i).sum()) for i, n in enumerate(names)}
    print(json.dumps(counts, indent=1))
    print(f"{len(assign)} items categorised -> {CATEGORIES_FILE}; median margin {np.median(margin):.3f}")


if __name__ == "__main__":
    main()
