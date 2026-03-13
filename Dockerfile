# ============================================================
# Ctrl-World: A Controllable Generative World Model
# Base: CUDA 12.6 + cuDNN 9 + Ubuntu 22.04
# Python 3.11, PyTorch 2.7.1+cu126
# ============================================================
FROM nvidia/cuda:12.6.3-cudnn9-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/workspace/models \
    HF_HOME=/workspace/models/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    git \
    git-lfs \
    curl \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch 2.7.1 with CUDA 12.6
RUN pip install \
    torch==2.7.1+cu126 \
    torchvision==0.22.1+cu126 \
    --index-url https://download.pytorch.org/whl/cu126

# Install core ML dependencies (pinned versions matching current environment)
RUN pip install \
    diffusers==0.34.0 \
    transformers==4.53.2 \
    accelerate==1.12.0 \
    numpy==2.4.2 \
    scipy==1.17.0 \
    pandas==3.0.0 \
    einops==0.8.2 \
    tqdm==4.67.3 \
    mediapy==1.2.6

# Install logging and tracking tools
RUN pip install \
    wandb==0.24.2 \
    swanlab==0.7.7

# Install video/data processing tools
RUN pip install \
    decord==0.6.0 \
    torchcodec==0.10.0

# Install JAX with CUDA 12 support (for pi0.5 integration)
RUN pip install \
    "jax[cuda12]==0.5.3" \
    jaxlib==0.5.3 \
    flax==0.10.2

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Install openpi as editable package (if openpi directory is present)
RUN if [ -d "/workspace/openpi" ]; then \
        pip install uv==0.10.0 && \
        cd /workspace/openpi && \
        GIT_LFS_SKIP_SMUDGE=1 pip install -e . && \
        GIT_LFS_SKIP_SMUDGE=1 pip install -e packages/openpi-client; \
    fi

# Create necessary directories for data and model checkpoints
RUN mkdir -p /workspace/models /workspace/dataset_example /workspace/synthetic_traj

# Default command
CMD ["/bin/bash"]
