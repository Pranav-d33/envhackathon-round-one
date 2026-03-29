# SRE Incident Response — OpenEnv Environment
# Dockerfile for Hugging Face Spaces
# Build: docker build -t sre-incident-env .
# Run:   docker run -p 7860:7860 sre-incident-env

FROM python:3.11-slim

LABEL maintainer="SRE Incident Response Environment"
LABEL description="OpenEnv SRE Incident Response RL Environment"
LABEL org.opencontainers.image.title="sre-incident-response"
LABEL org.opencontainers.image.version="1.0.0"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY openenv.yaml .
COPY baseline.py .

# Ensure correct ownership
RUN chown -R appuser:appuser /app

USER appuser

# HF Spaces uses port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
