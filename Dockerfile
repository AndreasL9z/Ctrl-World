# ============================================================
# Ctrl-World: A Controllable Generative World Model
# Base: CUDA 12.6.3 + cuDNN (devel) + Ubuntu 22.04
# Python 3.11, PyTorch 2.7.1 (via conda, matches local env)
# ============================================================
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH \
    TORCH_HOME=/workspace/models \
    HF_HOME=/workspace/models/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    git \
    git-lfs \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    conda clean -afy

# Create conda environment with Python 3.11
RUN conda create -n ctrl-world python=3.11 -y && \
    conda clean -afy

# Install PyTorch 2.7.1 with CUDA 12.6 via conda (matches local environment exactly)
RUN conda run -n ctrl-world conda install -y \
    pytorch==2.7.1 \
    torchvision \
    pytorch-cuda=12.6 \
    -c pytorch -c nvidia && \
    conda clean -afy

# Install core ML dependencies via pip inside the conda env
RUN conda run -n ctrl-world pip install --no-cache-dir \
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
RUN conda run -n ctrl-world pip install --no-cache-dir \
    wandb==0.24.2 \
    swanlab==0.7.7

# Install video/data processing tools
RUN conda run -n ctrl-world pip install --no-cache-dir \
    decord==0.6.0

# Install JAX with CUDA 12 support (for pi0.5 integration)
RUN conda run -n ctrl-world pip install --no-cache-dir \
    "jax[cuda12]==0.5.3" \
    flax==0.10.2

# Make conda env activation automatic in bash
RUN echo "conda activate ctrl-world" >> /root/.bashrc
ENV PATH=/opt/conda/envs/ctrl-world/bin:$PATH

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Install openpi as editable package (if openpi directory is present)
RUN if [ -d "/workspace/openpi" ]; then \
        conda run -n ctrl-world pip install uv==0.10.0 && \
        cd /workspace/openpi && \
        GIT_LFS_SKIP_SMUDGE=1 conda run -n ctrl-world pip install -e . && \
        GIT_LFS_SKIP_SMUDGE=1 conda run -n ctrl-world pip install -e packages/openpi-client; \
    fi

# Create necessary directories
RUN mkdir -p /workspace/models /workspace/dataset_example /workspace/synthetic_traj

# Default command
CMD ["/bin/bash", "--login"]
