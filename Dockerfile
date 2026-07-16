# Learn2Clean — Dockerfile
# Fully self-contained: includes TabPFN weights + OpenML dataset cache
#
# Build:  docker build -t learn2clean .
# Run:    docker run --rm -v $(pwd)/results:/app/results learn2clean python reproduce_table2.py --skip-rl --skip-oracle --skip-il --skip-cirl
# Shell:  docker run --rm -it -v $(pwd)/results:/app/results learn2clean bash

FROM python:3.13-slim

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git build-essential \
    libopenblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Install Poetry ─────────────────────────────────────────────────────────────
ENV POETRY_VERSION=2.4.1 \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies (cached layer) ────────────────────────────────
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --without dev

# ── Install extra packages (tabpfn + openml not in pyproject.toml) ────────────
RUN pip install --no-cache-dir tabpfn openml

# ── Copy TabPFN pre-authorized model weights ───────────────────────────────────
# NOTE: tabpfn_cache/ is not in the repo (203MB, too large for GitHub).
# TabPFN will download weights on first run — requires internet access.
# To pre-bundle weights: copy ~/Library/Caches/tabpfn/ to tabpfn_cache/ locally
# then rebuild. The weights are cached at /root/.cache/tabpfn/ inside the container.
# COPY tabpfn_cache/ /root/.cache/tabpfn/

# ── Copy all source code ──────────────────────────────────────────────────────
COPY src/ ./src/
COPY Learn2Clean_TFM/ ./Learn2Clean_TFM/
COPY il/ ./il/
COPY data/ ./data/
COPY experiments/ ./experiments/
COPY reproduce_table2.py quick_test.py README.md ./

# ── Copy OpenML dataset cache ─────────────────────────────────────────────────
# datasets/ contains *_raw.parquet files; openml_loader expects outputs/datasets/
COPY datasets/ ./datasets/
RUN mkdir -p /app/outputs/datasets && \
    cp /app/datasets/*_raw.parquet /app/outputs/datasets/ 2>/dev/null || true

# ── Create .env (WandB offline, no API key needed) ────────────────────────────
RUN printf 'WANDB_API_KEY=offline\nWANDB_MODE=offline\n' > /app/.env

# ── Install the project package ───────────────────────────────────────────────
RUN poetry install --only-root

# ── Environment variables ──────────────────────────────────────────────────────
ENV PYTHONPATH=/app/src:/app \
    WANDB_MODE=offline \
    PROJECT_ROOT=/app \
    XDG_CACHE_HOME=/root/.cache

# ── Create output directories ─────────────────────────────────────────────────
RUN mkdir -p /app/results /app/il/checkpoints /app/outputs/datasets

# ── Verify setup on build ─────────────────────────────────────────────────────
COPY docker_verify.py ./
RUN python docker_verify.py

# ── Default command ────────────────────────────────────────────────────────────
CMD ["python", "reproduce_table2.py", \
     "--skip-rl", "--skip-oracle", "--skip-il", "--skip-cirl"]
