# Use the official Python 3.10 slim image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app
RUN pip install --upgrade pip
# Install system dependencies needed for torchaudio and other packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libsndfile1 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to cache the pip installs
COPY requirements.txt .

# CRITICAL: Install CPU-only PyTorch first to save space and prevent CUDA errors
RUN pip install --default-timeout=1000 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the requirements with increased timeout
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Create the reports folder with correct permissions so the app can write PDFs
RUN mkdir -p /app/reports && chmod 777 /app/reports

# Copy the rest of your application code into the container
COPY . .

# Expose the specific port Hugging Face requires
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:7860/docs || exit 1

# Command to run the FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]