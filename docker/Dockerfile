FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Placeholder Dockerfile - THIS FILE IS A STARTING POINT.
# The final Dockerfile will install Python, PyTorch (matching CUDA), pyigtl,
# and your model weights. Build and test on the target system (WSL2 or Linux).

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip && pip install -r /app/requirements.txt

COPY src/ /app/src/

CMD ["python3", "src/service/inference_service.py"]
