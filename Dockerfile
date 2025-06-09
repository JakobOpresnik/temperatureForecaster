# ---------- Stage 1: Builder ----------
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y curl build-essential && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

WORKDIR /app

COPY pyproject.toml poetry.lock* /app/

# Install dependencies to cache build layers and speed up final install
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-root --no-interaction --no-ansi && \
    rm -rf /root/.cache

COPY src/ /app/src/
COPY params.yaml /app/

# ---------- Stage 2: Runtime ----------
FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl build-essential && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

WORKDIR /app

COPY pyproject.toml poetry.lock* /app/
COPY src/ /app/src/
COPY params.yaml /app/

# Install dependencies fresh in runtime (no copying of site-packages)
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-root --no-interaction --no-ansi && \
    rm -rf /root/.cache

WORKDIR /app/src

EXPOSE 8000

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
