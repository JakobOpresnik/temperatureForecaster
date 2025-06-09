# ---------- Stage 1: Builder ----------
FROM python:3.11-slim as builder

WORKDIR /app

# Install build tools + Poetry
RUN apt-get update && apt-get install -y curl build-essential && rm -rf /var/lib/apt/lists/*
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Copy pyproject + lock and generate requirements.txt
COPY pyproject.toml poetry.lock* /app/
RUN poetry export -f requirements.txt --only main --without-hashes -o requirements.txt

# ---------- Stage 2: Runtime ----------
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and config
COPY src/serve.py /app/src/
COPY params.yaml /app/

EXPOSE 8000

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
