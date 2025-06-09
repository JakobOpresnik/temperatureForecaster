# ---------- Stage 1: Builder ----------
FROM python:3.11-slim-bullseye AS builder

# Install system dependencies for building and running your Python libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt-dev \
    zlib1g-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

WORKDIR /app

COPY pyproject.toml poetry.lock* /app/

# Install only main dependencies, no dev
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-root --no-interaction --no-ansi && \
    rm -rf /root/.cache

# ---------- Stage 2: Runtime ----------
FROM python:3.11-slim-bullseye

RUN apt-get update && apt-get install -y --no-install-recommends \
    libffi7 \
    libssl1.1 \
    libxml2 \
    libxslt1.1 \
    zlib1g \
    libblas3 \
    liblapack3 \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/serve.py /app/src/
COPY params.yaml /app/

EXPOSE 8000

WORKDIR /app/src

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]

