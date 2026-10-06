# Niche API container. CPU-only torch keeps the image ~1.5 GB instead of ~6 GB.
# The precomputed index (embeddings, items, categories) is downloaded at startup from the GitHub
# release named in NICHE_DATA_RELEASE, so a deploy never has to re-embed 45k images on a CPU.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl gzip \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as uid 1000; make everything writable for it.
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-deploy.txt

COPY app app
COPY web web
COPY start.sh .
COPY data/brands.yaml data/known_brands.yaml data/traits_vocab.yaml data/traits_vocab_learned.json \
     data/brand_traits.json data/brand_tiers.json data/
RUN chown -R user:user /app
USER user

ENV PORT=7860 \
    HF_HOME=/tmp/hf \
    NICHE_DATA_RELEASE=https://github.com/tanisha-jainn/niche/releases/download/data-v1
EXPOSE 7860
CMD ["sh", "start.sh"]
