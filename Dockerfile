# ---------- Stage 1: Builder ----------
FROM python:3.11-slim AS builder

WORKDIR /app

# Install Poetry
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=2.1.3 python3 -
ENV PATH="/root/.local/bin:$PATH"

# Copy pyproject and lockfile
COPY pyproject.toml poetry.lock* /app/

# Export only runtime dependencies to requirements.txt
RUN poetry export --only main --without-hashes -f requirements.txt -o requirements.txt

# ---------- Stage 2: Runtime ----------
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only needed application code
COPY src/serve.py /app/src/
COPY params.yaml /app/

# Clean MLflow and other caches that might cause image bloat
RUN rm -rf /root/.cache /root/.mlflow /root/.cache/torch /root/.cache/huggingface

EXPOSE 8000
WORKDIR /app/src

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
