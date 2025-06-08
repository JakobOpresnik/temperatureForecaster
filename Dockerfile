FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system packages in one layer (clean cache to reduce size)
RUN apt-get update && apt-get install -y \
    curl build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Copy only dependency files first (for better cache)
COPY pyproject.toml poetry.lock* ./

# Install dependencies without dev packages
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi

# Now copy the rest of the code
COPY . .

# Default command
CMD ["python", "serve.py"]
