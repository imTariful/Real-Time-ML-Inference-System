FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy scripts and dummy model generation logic
# We need the models directory to exist so COPY doesn't fail if it's empty
COPY scripts/ scripts/
COPY models/ models/
# Generate models inside the image
RUN python scripts/create_dummy_model.py

FROM python:3.9-slim

WORKDIR /app

# Install curl for potential healthchecks (though K8s probes use internal connection usually)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser

# Copy installed packages
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy generated models
COPY --from=builder /app/models /app/models

# Copy code
COPY . .

# Fix permissions
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
