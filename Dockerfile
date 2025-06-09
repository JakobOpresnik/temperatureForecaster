# Stage 1: builder
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y curl build-essential && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=2.1.3 python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

WORKDIR /app

COPY pyproject.toml poetry.lock* /app/

RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-root --no-interaction --no-ansi && \
    poetry export -f requirements.txt --without-hashes --only main -o requirements.txt && \
    rm -rf /root/.cache

COPY src/ /app/src/
COPY params.yaml /app/

# Stage 2: runtime
FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/requirements.txt /app/
COPY --from=builder /app/src /app/src
COPY --from=builder /app/params.yaml /app/

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app/src

EXPOSE 8000

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
