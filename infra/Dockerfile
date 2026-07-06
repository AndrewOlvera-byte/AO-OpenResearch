# Base image already ships a Blackwell-capable PyTorch (CUDA 12.8, sm_120),
# so there is no torch re-install / version churn during the build like the
# previous cu124 base caused on the RTX 5070 Ti.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_ALLOW_TF32_CUBLAS=1 \
    CUDA_MODULE_LOADING=LAZY \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# gym-microrts drives a Java MicroRTS engine through JPype, so a JDK is required.
# xvfb/libgl are only needed if you render; kept headless-friendly but available.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates wget unzip \
      build-essential g++ gcc \
      openjdk-17-jdk-headless \
      xvfb libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# gym-microrts pins gym 0.21, whose setup.py is rejected by modern setuptools
# (the `extras_require` marker error). Pin the build toolchain and skip build
# isolation so gym builds against the compatible setuptools.
RUN pip install "pip<24.1" "setuptools==65.5.0" "wheel==0.38.4"

# Install dependencies first for Docker layer caching.
COPY pyproject.toml /workspace/pyproject.toml
RUN pip install --no-build-isolation -e .

# Copy project code (editable install above already wired up dependencies).
COPY . /workspace

# Keep container running for interactive CLI access.
CMD ["tail", "-f", "/dev/null"]
