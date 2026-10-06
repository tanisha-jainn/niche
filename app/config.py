import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=value lines, shell env wins over the file."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("\"'")
        if key and value:
            os.environ.setdefault(key, value)


_load_dotenv(ROOT / ".env")
DATA = ROOT / "data"
WEB = ROOT / "web"

BRANDS_FILE = DATA / "brands.yaml"              # seed list of indie stores to crawl
KNOWN_BRANDS_FILE = DATA / "known_brands.yaml"  # mainstream brands + aesthetic descriptions
CATALOG_FILE = DATA / "catalog.jsonl"           # raw crawl output
CRAWL_REPORT_FILE = DATA / "crawl_report.json"
INDEX_FILE = DATA / "index.jsonl"               # products that were successfully embedded
EMBEDDINGS_FILE = DATA / "embeddings.npy"       # aligned row-for-row with INDEX_FILE
IMAGE_CACHE = DATA / "images"
UPLOADS = DATA / "uploads"
PROFILE_FILE = DATA / "profile.json"
BRAND_DESC_CACHE = DATA / "brand_descriptions.json"

# Fashion-tuned CLIP; falls back to a generic LAION CLIP if the hub download fails.
CLIP_MODEL = os.environ.get("CLIP_MODEL", "hf-hub:Marqo/marqo-fashionCLIP")
CLIP_FALLBACK = ("ViT-B-32", "laion2b_s34b_b79k")

IMAGE_WIDTH = 512
MAX_PRODUCTS_PER_BRAND = int(os.environ.get("MAX_PRODUCTS_PER_BRAND", "120"))

CLAUDE_MODEL = "claude-opus-5"
