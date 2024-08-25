# Use an official NVIDIA CUDA base image with PyTorch pre-installed
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install necessary Python packages
RUN pip3 install fastapi uvicorn transformers torch==2.0.1+cu118 requests --extra-index-url https://download.pytorch.org/whl/cu118

# Copy the application code
COPY ./app /app

# Set the working directory
WORKDIR /app

# Copy the performance test script
COPY ./performance_test.py /app/performance_test.py

# Expose the application port
EXPOSE 8000

# Run the FastAPI app with Uvicorn and GPU support
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
