# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.11

# ---------------------------------------------------------------------------
# Shared OS layer: compilers + MPI so wheels that need them never require
# a host-side apt/pip install.
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libopenmpi-dev \
        openmpi-bin \
        libfftw3-dev \
        liblapack-dev \
        libblas-dev \
        libhwloc-dev \
        libnuma-dev \
        numactl \
        rsync \
        procps \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE.md ./
COPY src/ ./src/
COPY completions/ ./completions/

RUN pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# Runtime: core + HPC extras. `forge` is on PATH; no host pip needed.
# ---------------------------------------------------------------------------
FROM base AS runtime

RUN pip install ".[hpc]"

RUN useradd -m -u 1000 forge && chown -R forge:forge /app
USER forge

ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    KMP_AFFINITY=granularity=fine,compact,1,0 \
    PATH="/home/forge/.local/bin:${PATH}"

WORKDIR /work
CMD ["forge", "--help"]

# ---------------------------------------------------------------------------
# Dev: runtime + pytest/ruff/mypy and the test tree baked into the image.
# ---------------------------------------------------------------------------
FROM base AS dev

COPY tests/ ./tests/

RUN pip install ".[dev,hpc]"

ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    PYTEST_ADDOPTS="--tb=line"

WORKDIR /app
CMD ["python", "-m", "pytest", "-q"]
