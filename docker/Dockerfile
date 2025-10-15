FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Install system packages and Python
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ca-certificates curl build-essential \
    libsm6 libxext6 libxrender1 ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch wheels for CUDA 12.1 (suitable for modern RTX 40xx GPUs).
# Adjust the torch+vision versions if you need a specific release.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    "torch" "torchvision"

# Copy requirements and install the remaining Python deps. Note: torch and
# torchvision are already installed above via the CUDA wheel index.
COPY requirements.txt /app/requirements.txt
RUN sed -n '1,200p' /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source
COPY src/ /app/src/

EXPOSE 18945/tcp

ENTRYPOINT ["python3", "src/service/inference_service.py"]
