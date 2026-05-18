FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy Pipfile and install dependencies
COPY Pipfile Pipfile.lock ./
RUN pip install pipenv && \
    pipenv install --system --deploy

# Copy application code
COPY . .

# Expose port
EXPOSE 11434

# Run the application with hot reload
CMD ["python", "run.py"]
