FROM python:3.11-slim

WORKDIR /app

# 1. Install system dependencies (minimal set, no unnecessary layers)
RUN apt-get update && apt-get install -y \
  git curl && \
  rm -rf /var/lib/apt/lists/*

# 2. Install Poetry
RUN pip install --no-cache-dir poetry==1.3.1

# 3. Copy only dependency files first to leverage Docker cache
COPY pyproject.toml poetry.lock* ./

# 4. Install dependencies (cached if pyproject.toml/poetry.lock didn't change)
RUN poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction --no-ansi

# 5. Now copy your actual source code
COPY src/ ./src

# 6. Set required env variables (optional: use `.env` if many)
ENV MLFLOW_TRACKING_URI=https://dagshub.com/JakobOpresnik/temperatureForecaster.mlflow
ENV MLFLOW_REGISTRY_URI=https://dagshub.com/JakobOpresnik/temperatureForecaster.mlflow

# 7. Expose port and run the FastAPI app
EXPOSE 8000
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]