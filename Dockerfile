FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    git curl && \
    rm -rf /var/lib/apt/lists/*

# Install poetry and project dependencies
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry config virtualenvs.create false && poetry install --no-dev

# Copy source code
COPY src/ ./src

# Set env vars for MLflow tracking
ENV MLFLOW_TRACKING_URI=https://dagshub.com/JakobOpresnik/temperatureForecaster.mlflow
ENV MLFLOW_REGISTRY_URI=https://dagshub.com/JakobOpresnik/temperatureForecaster.mlflow

# Run the app
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]