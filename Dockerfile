# -------- Stage 1: Builder --------
FROM python:3.11-slim AS builder

# System deps for poetry & builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

WORKDIR /app

# Copy only poetry files first to leverage Docker cache
COPY pyproject.toml poetry.lock* /app/

# Install runtime dependencies only
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi --no-root && \
    rm -rf /root/.cache

# -------- Stage 2: Runtime --------
FROM python:3.11-slim

WORKDIR /app

# Copy only installed packages from builder (no dev, clean)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app source code and config
COPY src/serve.py /app/src/
COPY params.yaml /app/

# Final cleanup
RUN apt-get purge -y && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/*

EXPOSE 8000

WORKDIR /app/src

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]