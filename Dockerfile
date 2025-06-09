# Use Python 3.11 base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy lock and project file first to leverage Docker cache
COPY poetry.lock pyproject.toml /app/

# Install Poetry (v2.1.3 like yours)
RUN pip install poetry==2.1.3

# Disable virtualenv creation and install dependencies
RUN poetry config virtualenvs.create false && poetry install --no-root --no-interaction --no-ansi

# Copy only what’s needed for running the app
COPY src/serve.py /app/src/
COPY params.yaml /app/

# Expose port 8000 (Uvicorn/FastAPI default)
EXPOSE 8000

# Set working directory to app source
WORKDIR /app/src

# Run Uvicorn
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
