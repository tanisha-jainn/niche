#!/bin/sh
# Fetch the precomputed index if this container doesn't have it yet, then serve.
set -e
cd "$(dirname "$0")"
mkdir -p data/uploads data/profiles
for f in index.jsonl.gz embeddings.npy categories.npz; do
  if [ ! -f "data/$f" ]; then
    echo "downloading $f from $NICHE_DATA_RELEASE"
    curl -fsSL "$NICHE_DATA_RELEASE/$f" -o "data/$f"
  fi
done
[ -f data/index.jsonl ] || gunzip -k data/index.jsonl.gz
exec uvicorn app.server:app --host 0.0.0.0 --port "${PORT:-7860}"
