#!/bin/sh
# Fetch the precomputed index + ONNX models if missing, then serve.
# `sh start.sh --fetch-only` downloads and exits (used at image build time).
set -e
cd "$(dirname "$0")"
mkdir -p data/uploads data/profiles
for f in index.jsonl.gz embeddings.npy categories.npz \
         clip_text.onnx clip_text.onnx.data clip_image.onnx clip_image.onnx.data; do
  if [ ! -f "data/$f" ]; then
    echo "downloading $f"
    python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
      "$NICHE_DATA_RELEASE/$f" "data/$f"
  fi
done
[ -f data/index.jsonl ] || python -c "import gzip, shutil; shutil.copyfileobj(gzip.open('data/index.jsonl.gz'), open('data/index.jsonl', 'wb'))"
[ "$1" = "--fetch-only" ] && exit 0
exec uvicorn app.server:app --host 0.0.0.0 --port "${PORT:-10000}"
