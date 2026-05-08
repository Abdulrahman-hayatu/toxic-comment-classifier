# This Dockerfile sets up a container for a FastAPI app that uses a Hugging Face model.
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download models from Hugging Face Hub
RUN pip install --no-cache-dir huggingface-hub && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('Abdulrahman-Hayatu/toxic-comment-classifier', local_dir='/app/models')"

# Copy source code
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

# Set PYTHONPATH to include the app directory so we can import modules from src and api
ENV PYTHONPATH=/app
ENV PORT=7860

# The port our FastAPI app listens on
ENV PORT=7860

# Start the server
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]