# Use a base image with Python + CUDA support
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set up environment
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip sox ffmpeg git curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Ollama (LLaMA API)
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the application code
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
