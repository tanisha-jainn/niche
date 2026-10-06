"""Claude helpers. Every call is metered (tokens + dollars) into data/llm_usage.jsonl and refused once
the budget is spent. Everything degrades to None when no key is set, so the app never breaks on it."""
import base64
import io
import json
import os
import time

from PIL import Image

from .config import DATA

USAGE_FILE = DATA / "llm_usage.jsonl"
BUDGET_USD = float(os.environ.get("NICHE_LLM_BUDGET_USD", "5"))
# Sonnet 5 for reads (user-approved 2026-10-04): same prompts at 40% of Opus 5's price.
MODEL = os.environ.get("NICHE_MODEL", "claude-sonnet-5")
# $ per million tokens: input, output, cache read, cache write (Anthropic first-party rates)
PRICES = {
    "claude-opus-5": (5.0, 25.0, 0.5, 6.25),
    "claude-sonnet-5": (2.0, 10.0, 0.2, 2.5),
    "claude-haiku-4-5": (1.0, 5.0, 0.1, 1.25),
}
IMAGE_MAX_PX = 512          # ~300 image tokens; trait reading doesn't need more
MAX_IMAGES_PER_CALL = 8
MAX_TRAITS_PER_BATCH = 6
MAX_TRAITS_PER_BRAND = 4

TRAIT_STYLE = (
    "Write each trait the way a shopper would describe clothes to a friend: the garment plus the one or two "
    "features that define it - cut, fabric, detail or colour - e.g. 'low-rise black flared pants', 'lace trim on "
    "straight-leg denim', 'cropped boxy cable knit'. Two to six words each. Each trait should be something a "
    "person could look for across many brands, not an inventory of one exact piece: 'bubble-hem midi dress' and "
    "'ruffle-trim spaghetti straps', not 'ruched bubble-hem midi dress in sandy beige with picot lace'. Never use "
    "scene, era or subculture labels: no words ending in -core, no decades (Y2K, 90s), no 'vibe', 'aesthetic', "
    "'chic', 'boho', 'grunge', 'preppy', 'streetwear', 'quiet luxury', 'it-girl', and no brand names."
)


class BudgetExceeded(RuntimeError):
    pass


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


# ---------- metering ----------

def usage_summary() -> dict:
    calls = input_tokens = output_tokens = 0
    cost = 0.0
    by_kind: dict[str, int] = {}
    if USAGE_FILE.exists():
        for line in open(USAGE_FILE):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            calls += 1
            input_tokens += r["input_tokens"] + r.get("cache_read", 0) + r.get("cache_write", 0)
            output_tokens += r["output_tokens"]
            cost += r["cost_usd"]
            by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
    return {"calls": calls, "input_tokens": input_tokens, "output_tokens": output_tokens,
            "cost_usd": round(cost, 4), "budget_usd": BUDGET_USD, "remaining_usd": round(BUDGET_USD - cost, 4),
            "by_kind": by_kind, "model": MODEL}


def budget_left() -> bool:
    return usage_summary()["remaining_usd"] > 0


def _record(kind: str, response) -> None:
    u = response.usage
    p_in, p_out, p_cr, p_cw = PRICES.get(response.model, PRICES.get(MODEL, PRICES["claude-sonnet-5"]))
    cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
    cost = (u.input_tokens * p_in + u.output_tokens * p_out + cache_read * p_cr + cache_write * p_cw) / 1e6
    with open(USAGE_FILE, "a") as f:
        f.write(json.dumps({"ts": time.time(), "kind": kind, "model": response.model,
                            "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                            "cache_read": cache_read, "cache_write": cache_write,
                            "cost_usd": round(cost, 6)}) + "\n")


def _create(kind: str, **kwargs):
    if not budget_left():
        raise BudgetExceeded(f"Claude budget of ${BUDGET_USD:.2f} is used up (NICHE_LLM_BUDGET_USD to raise it).")
    import anthropic
    client = anthropic.Anthropic()
    resp = client.beta.messages.create(
        model=MODEL,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "low"},
        **kwargs,
    )
    _record(kind, resp)
    return resp


def _text(response) -> str:
    if response.stop_reason == "refusal":
        return ""
    return "".join(b.text for b in response.content if b.type == "text").strip()


def _safe(fn):
    """Never let an LLM failure break the app; budget exhaustion is logged like any other failure."""
    try:
        return fn()
    except Exception as e:  # auth, network, budget, SDK version mismatch...
        print(f"[llm] {e.__class__.__name__}: {e}")
        return None


def _image_block(image_bytes: bytes) -> dict:
    """Downscale before sending - tokens scale with pixels."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((IMAGE_MAX_PX, IMAGE_MAX_PX))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=82)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.standard_b64encode(buf.getvalue()).decode()}}


def _parse_json(text: str):
    text = text.strip()
    for candidate in (text, text[text.find("["): text.rfind("]") + 1]):
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return data
        except ValueError:
            continue
    return [line.strip("-*• \t\"'") for line in text.splitlines() if line.strip()]


# ---------- traits ----------

def traits_from_images(images: list[bytes], captions: list[str] | None = None,
                       context: str = "screenshots one person saved because they like the clothes in them") -> list[dict] | None:
    """One call for a whole batch: <=6 profile-level traits, each with the image numbers that show it."""
    if not available() or not images:
        return None
    images = images[:MAX_IMAGES_PER_CALL]
    content = []
    for i, b in enumerate(images):
        label = f"Image {i + 1}" + (f" ({captions[i]})" if captions and i < len(captions) and captions[i] else "")
        content.append({"type": "text", "text": label + ":"})
        content.append(_image_block(b))
    content.append({"type": "text", "text": (
        f"These are {len(images)} {context}. List 3-6 traits that recur or stand out across them - the things "
        "this person would want more of - most defining first. Ignore app UI, captions, faces and background. "
        "For each trait give the image numbers that show it. Reply with a JSON array of objects like "
        '[{"trait": "low-rise flared jeans", "images": [1, 3]}] and nothing else.')})

    def go():
        resp = _create("traits_images", max_tokens=400, system=TRAIT_STYLE,
                       messages=[{"role": "user", "content": content}])
        out = []
        for d in _parse_json(_text(resp)):
            if isinstance(d, str):
                out.append({"text": d, "images": []})
            elif isinstance(d, dict) and d.get("trait"):
                nums = [int(x) - 1 for x in d.get("images", []) if str(x).isdigit()]
                out.append({"text": str(d["trait"]), "images": [n for n in nums if 0 <= n < len(images)]})
        return out[:MAX_TRAITS_PER_BATCH] or None

    return _safe(go)


def polish_traits(candidates: list[str], nearest_titles: list[str]) -> list[str] | None:
    """Text-only (no image tokens): tidy the free vocabulary read into <=6 natural traits, using the titles
    of the catalog pieces the images sat closest to as extra evidence. ~300 input tokens."""
    if not available():
        return None
    msg = (
        "A free local matcher described someone's saved screenshots with these candidate traits:\n- "
        + "\n- ".join(candidates)
        + "\n\nThe catalog pieces most similar to their screenshots are titled:\n- "
        + "\n- ".join(nearest_titles[:12])
        + "\n\nRewrite this into 3-6 traits that best describe what this person keeps coming back to: merge "
          "near-duplicates, drop anything the titles contradict, add anything obvious the candidates missed. "
          "Reply with a JSON array of strings and nothing else."
    )

    def go():
        resp = _create("polish", max_tokens=200, system=TRAIT_STYLE, messages=[{"role": "user", "content": msg}])
        return [str(x) for x in _parse_json(_text(resp)) if isinstance(x, str)][:MAX_TRAITS_PER_BATCH] or None

    return _safe(go)


def traits_from_brand(name: str, titles: list[str], types: list[str], description: str | None = None) -> list[str] | None:
    """3-4 recurring traits of what a brand sells, from a sample of its catalog (or a description)."""
    if not available():
        return None
    parts = [f"Brand: {name}."]
    if titles:
        parts.append("Sample product titles:\n- " + "\n- ".join(titles))
    if types:
        parts.append("Product types: " + ", ".join(types))
    if description:
        parts.append("Description: " + description)
    parts.append("List 3-4 recurring traits of what this brand sells - the things a fan of the brand keeps "
                 "coming back for, most defining first. Reply with a JSON array of strings and nothing else.")

    def go():
        resp = _create("traits_brand", max_tokens=200, system=TRAIT_STYLE,
                       messages=[{"role": "user", "content": "\n\n".join(parts)}])
        return [str(x) for x in _parse_json(_text(resp)) if isinstance(x, str)][:MAX_TRAITS_PER_BRAND] or None

    return _safe(go)
