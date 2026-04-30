# Start from an official Python image
# 'slim' variant is smaller for fast deployments
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first Docker caches layers
# If requirements haven't changed, this layer is reused (faster builds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download models from Hugging Face Hub
RUN pip install --no-cache-dir huggingface-hub && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('Abdulrahman-Hayatu/toxic-comment-classifier-models', local_dir='/app/models')"

# Copy source code
COPY src/ ./src/
COPY api/ ./api/

# Set Python path so imports work
ENV PYTHONPATH=/app

# The port our FastAPI app listens on
# (Render and other platforms will set the PORT env variable)
ENV PORT=8000

# Start the server
# Use 0.0.0.0 (not localhost) so the container accepts external connections
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]