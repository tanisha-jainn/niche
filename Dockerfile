# Niche API. No torch: the CLIP encoders run as small ONNX models, so the image is ~450 MB and the
# server peaks under 512 MB of memory (fits free hosting tiers).
# The precomputed index and the models are baked into the image at build time from the GitHub release,
# so a cold start doesn't re-download anything.
FROM python:3.12-slim

WORKDIR /app
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

ARG NICHE_DATA_RELEASE=https://github.com/tanisha-jainn/niche/releases/download/data-v1
ENV NICHE_DATA_RELEASE=$NICHE_DATA_RELEASE
COPY start.sh .
RUN mkdir -p data && sh start.sh --fetch-only

COPY app app
COPY web web
COPY data/brands.yaml data/known_brands.yaml data/traits_vocab.yaml data/traits_vocab_learned.json \
     data/traits_vocab_vectors.npz data/brand_traits.json data/brand_tiers.json data/

RUN useradd -m -u 1000 user && mkdir -p data/uploads data/profiles && chown -R user:user /app
USER user

ENV PORT=10000
EXPOSE 10000
CMD ["sh", "start.sh"]
