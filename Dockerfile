FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y curl build-essential && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy dependency files early to cache layer
COPY pyproject.toml poetry.lock* /app/

# Install only main dependencies (no dev, no root package)
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-root --no-interaction --no-ansi

# Copy application files needed at runtime
COPY serve.py params.yaml /app/

# Expose port used by Uvicorn
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]