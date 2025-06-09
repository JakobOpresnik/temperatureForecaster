FROM pytorch/pytorch:2.0.1-cpu

WORKDIR /app

# Install system deps if needed
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install Poetry 2.1.3 explicitly
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=2.1.3 python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Copy poetry files and install only non-torch dependencies
COPY pyproject.toml poetry.lock* /app/

RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi --no-root && \
    rm -rf /root/.cache

# Copy your app code
COPY src/serve.py /app/src/
COPY params.yaml /app/

EXPOSE 8000

WORKDIR /app/src

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
