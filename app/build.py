"""Download product images from data/catalog.jsonl and embed them with CLIP.

Incremental: embeddings are cached per image URL, so a rebuild after growing the brand list
only embeds the new products. Writes data/index.jsonl (products with a usable image) and
data/embeddings.npy (aligned rows).
Run: python -m app.build
"""
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from .config import CATALOG_FILE, DATA, EMBEDDINGS_FILE, IMAGE_CACHE, INDEX_FILE
from .crawl import UA
from .embed import Embedder

EMBED_CACHE = DATA / "embed_cache.npz"   # keys: image-url hashes; values: unit vectors


def url_key(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()


def cache_path(url: str) -> Path:
    return IMAGE_CACHE / (url_key(url) + ".jpg")


def download(client: httpx.Client, url: str) -> Path | None:
    path = cache_path(url)
    if path.exists():
        return path
    try:
        r = client.get(url)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        img.thumbnail((512, 512))
        img.save(path, "JPEG", quality=88)
        return path
    except Exception:
        return None


def load_cache() -> dict[str, np.ndarray]:
    if not EMBED_CACHE.exists():
        return {}
    z = np.load(EMBED_CACHE)
    return dict(zip(z["keys"].tolist(), z["vecs"]))


def save_cache(cache: dict[str, np.ndarray]) -> None:
    keys = list(cache)
    np.savez(EMBED_CACHE, keys=np.array(keys), vecs=np.stack([cache[k] for k in keys]).astype(np.float32))


def main() -> None:
    products = [json.loads(line) for line in open(CATALOG_FILE)]
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)
    cache = load_cache()

    todo = [p for p in products if url_key(p["image"]) not in cache]
    print(f"{len(products)} products, {len(products) - len(todo)} already embedded, {len(todo)} to do", flush=True)

    if todo:
        with httpx.Client(timeout=20, follow_redirects=True, headers={"User-Agent": UA}) as client, \
                ThreadPoolExecutor(max_workers=16) as ex:
            paths = list(ex.map(lambda p: download(client, p["image"]), todo))
        fresh = [(p, path) for p, path in zip(todo, paths) if path]
        print(f"downloaded {len(fresh)}/{len(todo)} images", flush=True)

        embedder = Embedder()
        print(f"embedding with {embedder.name} on {embedder.device}", flush=True)
        batch = 64
        for i in range(0, len(fresh), batch):
            chunk = fresh[i:i + batch]
            imgs = []
            for _, path in chunk:
                with Image.open(path) as im:
                    imgs.append(im.convert("RGB"))
            for (p, _), vec in zip(chunk, embedder.images(imgs)):
                cache[url_key(p["image"])] = vec
            if (i // batch) % 10 == 0 or i + batch >= len(fresh):
                print(f"embedded {min(i + batch, len(fresh))}/{len(fresh)}", flush=True)
        save_cache(cache)

    kept = [p for p in products if url_key(p["image"]) in cache]
    E = np.stack([cache[url_key(p["image"])] for p in kept]).astype(np.float16)   # Catalog upcasts at load
    np.save(EMBEDDINGS_FILE, E)
    with open(INDEX_FILE, "w") as f:
        for p in kept:
            f.write(json.dumps(p) + "\n")
    print(f"index: {len(kept)} items from {len({p['brand'] for p in kept})} brands, dim {E.shape[1]} -> {INDEX_FILE}")


if __name__ == "__main__":
    main()
