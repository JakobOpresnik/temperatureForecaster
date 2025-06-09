# ---------- Stage 1: Builder ----------
FROM python:3.11-slim AS builder

# Install system deps
RUN apt-get update && apt-get install -y curl build-essential && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

WORKDIR /app

# Copy and install dependencies
COPY pyproject.toml poetry.lock* /app/
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-root --no-interaction --no-ansi

# ---------- Stage 2: Runtime ----------
FROM python:3.11-slim

WORKDIR /app

# Copy only installed packages and app source files
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/src /app/src
COPY --from=builder /app/params.yaml /app/

# Clean up (optional but safe)
RUN apt-get purge -y && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/*

EXPOSE 8000

WORKDIR /app/src
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]