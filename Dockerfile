FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
  git curl && \
  rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.3.1

# Copy only if both files exist
COPY pyproject.toml ./

# Install project dependencies
RUN poetry config virtualenvs.create false && poetry install --no-dev --no-interaction --no-ansi

COPY src/ ./src

ENV MLFLOW_TRACKING_URI=https://dagshub.com/JakobOpresnik/temperatureForecaster.mlflow
ENV MLFLOW_REGISTRY_URI=https://dagshub.com/JakobOpresnik/temperatureForecaster.mlflow

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
