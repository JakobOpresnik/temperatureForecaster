FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
  git curl build-essential && \
  rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==1.3.1

COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction --no-ansi