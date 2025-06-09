# ---------- Stage 1: Builder ----------
FROM python:3.11-slim AS builder

# Install system deps needed for build & Poetry
RUN apt-get update && apt-get install -y curl build-essential && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* /app/

# Install dependencies without creating virtualenv (install in global env)
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-root --no-interaction --no-ansi && \
    rm -rf /root/.cache /root/.mlflow /root/.cache/torch /root/.cache/huggingface

# ---------- Stage 2: Runtime ----------
FROM python:3.11-slim

WORKDIR /app

# Copy only installed Python packages (site-packages) and binaries from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy your app source and config
COPY src/serve.py /app/src/
COPY params.yaml /app/

# Clean up package manager caches (safe to do in slim)
RUN apt-get purge -y && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/*

EXPOSE 8000

WORKDIR /app/src

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
