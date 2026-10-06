"""Read traits for every indexed brand and every known mainstream brand once, so naming a brand never
costs a Claude call at runtime. Skips brands already cached. Prints the spend when done.

Run: python -m app.precompute [--workers 4]
"""
import sys
from concurrent.futures import ThreadPoolExecutor

from . import llm, traits
from .index import Catalog
from .taste import known_brands


def main(workers: int = 4) -> None:
    catalog = Catalog()
    names = sorted(catalog.brand_rows) + sorted(known_brands())
    todo = [n for n in names if traits.cache_get(traits.BRAND_TRAITS_CACHE, n.lower()) is None]
    before = llm.usage_summary()
    print(f"{len(names)} brands, {len(names) - len(todo)} cached, {len(todo)} to read with {llm.MODEL}", flush=True)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for name, (_, found) in zip(todo, ex.map(lambda n: traits.from_brand(catalog, n), todo)):
            done += 1
            if not found:
                print(f"  ! nothing read for {name}", flush=True)
            if done % 25 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)}", flush=True)

    after = llm.usage_summary()
    print(f"spent ${after['cost_usd'] - before['cost_usd']:.3f} over {after['calls'] - before['calls']} calls "
          f"({after['input_tokens'] - before['input_tokens']:,} in / {after['output_tokens'] - before['output_tokens']:,} out); "
          f"total so far ${after['cost_usd']:.3f} of ${after['budget_usd']:.0f}")


if __name__ == "__main__":
    w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    main(w)
