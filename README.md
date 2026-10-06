# Niche

Search 400+ independent brands as one storefront. Relevance first, no sponsored results, in your size.

**Why:** searching for clothes means Google, and Google's first page is paid placements from the same
few brands. Small brands exist but never surface; finding one takes screenshot → reverse search →
forget. Niche indexes independent brands itself and ranks only by what you asked for.

**What you can do:** type what you want ("low-rise black pants", "fall jacket") → ranked results across
every brand we index, each with a swipeable gallery, price in your currency, and per-size stock; filter
to your size, by price, by garment type; open a piece for its full gallery, sizes, description, similar
pieces from other brands, and a link to buy at the brand. Save pieces; brands you touch collect under
*Brands → You've found* so you stop forgetting them. Optional personal lens: screenshots, a public
Pinterest board, or a 30-second "tap what you'd wear" quiz nudge results toward your taste and power
*Brands → For you*.

**How it works**

0. **Search** (`app/search.py`) — CLIP text→image similarity for meaning + BM25 over titles/types/tags
   for the literal words + a garment-category gate + phrase/colour bonuses; optional taste nudge applied
   *after* relevance; at most two pieces per brand per page; colourways collapsed; fees and placeholder
   "products" excluded. Sizes are normalised at index load (`app/sizes.py`), so rules can change
   without a recrawl.

1. **Brand index** — a seed list of ~460 indie stores (`data/brands.yaml`), researched across
   aesthetics and regions (streetwear, minimal, Y2K, romantic, sustainable, size-inclusive/modest,
   Asia/LatAm, CA/AU/UK, denim/workwear) and probed for a live storefront. Most run on Shopify, and
   every Shopify store exposes `/products.json` publicly, so one crawl yields a structured catalog
   with images, prices and stock — currently ~45k products from ~440 brands.
2. **One vector space** — every product photo, every screenshot you upload, and every brand you name
   is embedded with a fashion-tuned CLIP model, so "looks like this TikTok screenshot" and
   "feels like Aritzia" are the same kind of query.
3. **Taste read-back** — Claude reads each screenshot and each brand you name into 3–4 concrete
   traits in shopper language ("low-rise black flared pants", "lace trim on straight-leg denim") —
   never scene labels like Y2K or -core (those are rejected in code). You confirm, edit, reweight or
   untick them before anything is recommended.
4. **Brands, not pieces** — each confirmed trait is a text query over the catalog; a brand "carries"
   a trait when its best pieces score close to the trait's best matches anywhere in the index. Brands
   are ranked by how many of your traits they cover. Every card says why — *"Carries X, Y and Z — 3
   of your 5 traits"* — with the piece that proves each trait. Brands you already shop are excluded.
5. **Spend control** — every Claude call is metered (tokens + dollars) to `data/llm_usage.jsonl`,
   shown in the header, capped by `NICHE_LLM_BUDGET_USD` (default $5). Brand and screenshot reads are
   cached so nothing is read twice. Put `ANTHROPIC_API_KEY=...` in `.env`.

## Run

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python -m app.crawl     # ~1 min: Shopify products.json for every seed brand
.venv/bin/python -m app.build     # downloads images, embeds with CLIP (first run downloads the model)
.venv/bin/uvicorn app.server:app --reload
```

Open http://127.0.0.1:8000. Optional: `export ANTHROPIC_API_KEY=...` before starting the server.

## Layout

```
app/crawl.py    Shopify crawler          -> data/catalog.jsonl
app/build.py    image download + CLIP    -> data/index.jsonl, data/embeddings.npy
app/embed.py    CLIP wrapper (image + text)
app/index.py    in-memory catalog
app/traits.py   trait extraction (via Claude), guardrails, caching, profile merge
app/recommend.py brand scoring by trait coverage, evidence, justification
app/taste.py    profile persistence (+ the older item-feed scorer)
app/llm.py      Claude calls, metering, budget cap
app/server.py   FastAPI routes
web/index.html  UI: inputs -> traits -> brand cards
data/brands.yaml        seed stores to crawl
data/known_brands.yaml  mainstream brands + descriptions used as text anchors
```

## Growing the brand list

```bash
.venv/bin/python -m app.grow candidates.jsonl   # probes each domain's /products.json; only live stores are added
.venv/bin/python -m app.crawl && .venv/bin/python -m app.build   # build is incremental - only new images get embedded
```

Candidate lines look like `{"name": "...", "domain": "brand.com", "currency": "USD", "tags": ["..."]}`.
Rejected domains (not Shopify, bot-blocked, dead) land in `data/grow_rejects.json`.

`python -m app.audit` shows, for each mainstream brand a user might name, how many catalog items
match strongly and from how many brands — the aesthetics at the top of that list are where the
seed list is thinnest and where to research next.

## Tuning knobs

- Add brands to `data/brands.yaml` (or via `app.grow`) and rerun crawl + build.
- `CLIP_MODEL` env var (default `hf-hub:Marqo/marqo-fashionCLIP`).
- Scoring weights and the per-brand cap live in `app/taste.py`.
