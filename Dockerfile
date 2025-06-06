FROM python:3.11-slim

WORKDIR /app

# Install Poetry via curl, then remove curl to save space
RUN apt-get update && apt-get install -y curl && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    apt-get purge -y curl && \
    rm -rf /var/lib/apt/lists/*

# Add Poetry to path
ENV PATH="/root/.local/bin:$PATH"

# Copy dependencies
COPY pyproject.toml ./
# (only copy poetry.lock if it exists and is committed)
# COPY poetry.lock ./

RUN poetry config virtualenvs.create false && poetry install --no-dev

# Copy application
COPY src/ ./src

# Set MLflow tracking env vars
ENV MLFLOW_TRACKING_URI=https://dagshub.com/JakobOpresnik/temperatureForecaster.mlflow
ENV MLFLOW_REGISTRY_URI=https://dagshub.com/JakobOpresnik/temperatureForecaster.mlflow

EXPOSE 8000

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
