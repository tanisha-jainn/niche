"""Approximate size normalisation. Every brand labels sizes differently (XS-XL, US 0-14, UK 6-18, EU 34-46,
waist 24-32, 1-2-3, "S/M", "One Size"); the filter works on one canonical letter scale, inferred per product
from the whole set of sizes it offers and the brand's country. Approximate by design - the user asked for
"approximate" and the brand's own size chart is one click away."""
import re

LETTERS = ["XXS", "XS", "S", "M", "L", "XL", "XXL"]
ONE = "ONE"

ALIASES = {
    "xxs": "XXS", "2xs": "XXS", "xxsmall": "XXS", "xx-small": "XXS",
    "xs": "XS", "x-small": "XS", "xsmall": "XS", "extra small": "XS", "petite": "XS",
    "s": "S", "sm": "S", "small": "S",
    "m": "M", "md": "M", "med": "M", "medium": "M",
    "l": "L", "lg": "L", "large": "L",
    "xl": "XL", "x-large": "XL", "xlarge": "XL", "extra large": "XL", "1x": "XL",
    "xxl": "XXL", "2xl": "XXL", "xx-large": "XXL", "3xl": "XXL", "4xl": "XXL", "5xl": "XXL", "2x": "XXL", "3x": "XXL", "4x": "XXL",
    "os": ONE, "o/s": ONE, "one size": ONE, "onesize": ONE, "one-size": ONE, "free": ONE, "free size": ONE,
    "u": ONE, "uni": ONE, "universal": ONE, "default title": ONE, "default": ONE, "standard": ONE,
}
REGION = {"GBP": "UK", "AUD": "UK", "NZD": "UK", "EUR": "EU", "SEK": "EU", "DKK": "EU", "NOK": "EU", "CHF": "EU",
          "PLN": "EU", "USD": "US", "CAD": "US", "MXN": "US", "BRL": "US", "INR": "UK", "JPY": "JP", "KRW": "KR"}
NUMERIC = {
    "US": {0: "XS", 2: "XS", 4: "S", 6: "S", 8: "M", 10: "M", 12: "L", 14: "L", 16: "XL", 18: "XL", 20: "XXL", 22: "XXL", 24: "XXL"},
    "UK": {4: "XXS", 6: "XS", 8: "S", 10: "S", 12: "M", 14: "L", 16: "L", 18: "XL", 20: "XL", 22: "XXL", 24: "XXL"},
    "EU": {32: "XXS", 34: "XS", 36: "S", 38: "S", 40: "M", 42: "M", 44: "L", 46: "XL", 48: "XL", 50: "XXL"},
    "IT": {36: "XXS", 38: "XS", 40: "S", 42: "S", 44: "M", 46: "L", 48: "XL", 50: "XXL"},
}
NUMERIC["JP"] = NUMERIC["US"]
NUMERIC["KR"] = NUMERIC["US"]
WAIST = {22: "XXS", 23: "XXS", 24: "XS", 25: "XS", 26: "S", 27: "S", 28: "M", 29: "M", 30: "L", 31: "L", 32: "XL", 33: "XL", 34: "XXL", 35: "XXL", 36: "XXL"}
SCALE_123 = {0: "XS", 1: "S", 2: "M", 3: "L", 4: "XL", 5: "XXL"}   # brand-own "1/2/3" scales

BOTTOMS = re.compile(r"\b(jean|jeans|denim|pant|pants|trouser|trousers|short|shorts|skirt)\b", re.I)
SHOES = re.compile(r"\b(shoe|shoes|boot|boots|sandal|sandals|sneaker|sneakers|loafer|loafers|heel|heels|mule|mules|flat|flats|slipper)\b", re.I)
SIZE_OPTION = re.compile(r"size|taille|größe|grösse|talla|taglia|maat|storlek|sz", re.I)
REGION_TAG = re.compile(r"\b(us|uk|eu|au|it|fr|de|jp)\b", re.I)
NUM = re.compile(r"\d+(?:\.\d+)?")


def _letters(token: str) -> list[str] | None:
    t = token.strip().lower().replace("–", "-")
    t = re.sub(r"\s*\(.*?\)\s*", " ", t).strip()          # "XS (US 0-2)" -> "xs"
    t = re.sub(r"^(size|sz|taille)\s*", "", t)
    if t in ALIASES:
        return [ALIASES[t]]
    if "/" in t or "-" in t:                                # "s/m", "m-l", "xxs/uk6" (letter wins)
        parts = [p.strip() for p in re.split(r"[/-]", t)]
        found = [ALIASES[p] for p in parts if p in ALIASES]
        if found:
            return list(dict.fromkeys(found))
    return None


def _numbers(token: str) -> tuple[list[float], str | None]:
    """Numbers in a label (a range like "1/2" or "8-10" gives both) and an explicit region tag
    if present ("UK 10", "W27", "EU38")."""
    nums = [float(x) for x in NUM.findall(token)][:2]
    region = REGION_TAG.search(token)
    return nums, (region.group().upper() if region else None)


def normalise(labels: list[str], currency: str, title: str = "") -> dict[str, list[str]]:
    """Map each raw size label to canonical letters (possibly several, e.g. "S/M"). Unknown -> []."""
    if SHOES.search(title) and not BOTTOMS.search(title):
        return {lab: [] for lab in labels}
    region = REGION.get(currency, "US")
    out: dict[str, list[str]] = {}
    numeric: dict[str, tuple[list[float], str | None]] = {}
    for lab in labels:
        letters = _letters(lab)
        if letters:
            out[lab] = letters
            continue
        nums, tag = _numbers(lab)
        if nums:
            numeric[lab] = (nums, tag)
        else:
            out[lab] = []
    if numeric:
        values = [n for nums, _ in numeric.values() for n in nums]
        is_123 = max(values) <= 5 and min(values) <= 1 and len(set(values)) >= 2 and all(v == int(v) for v in values)
        is_waist = BOTTOMS.search(title) and all(22 <= v <= 36 for v in values)
        for lab, (nums, tag) in numeric.items():
            if is_123:
                table = SCALE_123
            elif is_waist or lab.lower().startswith("w"):
                table = WAIST
            elif tag == "AU":
                table = NUMERIC["UK"]
            else:
                table = NUMERIC.get(tag if tag in NUMERIC else region, NUMERIC["US"])
            out[lab] = list(dict.fromkeys(table[int(n)] for n in nums if int(n) in table))
    return out


def size_option_index(options: list[dict]) -> int | None:
    """Which Shopify option is the size (option1/2/3). Named match first, then the one whose values look like sizes."""
    for o in options or []:
        if SIZE_OPTION.search(o.get("name") or ""):
            return int(o.get("position", 1))
    for o in options or []:
        values = [str(v) for v in (o.get("values") or [])]
        if values and sum(1 for v in values if _letters(v) or NUM.search(v)) >= max(1, len(values) // 2):
            return int(o.get("position", 1))
    return None
