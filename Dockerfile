# Serves the FSSAI RAG QA API (app/api.py). CPU-only: this is a demo/
# portfolio deployment target, not a GPU-backed production service.
FROM python:3.12-slim

WORKDIR /app

# curl is only for the HEALTHCHECK below; slim images don't ship it.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .

# CPU-only torch build, installed from PyTorch's own CPU wheel index
# BEFORE the rest of requirements-api.txt -- pip's default index would
# otherwise resolve the much larger CUDA-enabled build on Linux, which
# this CPU-served container never uses.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 \
    && pip install --no-cache-dir -r requirements-api.txt

# Application code.
COPY app/ ./app/
COPY src/ ./src/
COPY eval/ ./eval/

# Pre-built Chroma vector store, baked into the image for demo simplicity --
# see README.md ("Deployment notes") and decisions.md for why this tradeoff
# was made instead of re-running ingestion at container startup.
COPY data/chroma/ ./data/chroma/

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# start-period gives the startup warm-up (loading the embedding model, BM25
# corpus, and cross-encoder reranker once -- see app/api.py's lifespan)
# time to finish before failed checks count against the container.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
