# =============================================
# Dockerfile for Snake AI Training (GPU + FastAPI)
# =============================================
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install system dependencies
COPY requirements.txt .

# create & activate venv
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir "setuptools<81"
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

# Healthcheck (optional but recommended)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s \
  CMD curl -f http://localhost:8000/ || exit 1

# Run the server
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]