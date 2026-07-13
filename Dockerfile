# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps for HPC/DFT
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gfortran libopenmpi-dev openmpi-bin \
    rsync numactl procps && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/
COPY completions/ ./completions/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Non-root user for security
RUN useradd -m -u 1000 forge && chown -R forge:forge /app
USER forge

ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    KMP_AFFINITY=granularity=fine,compact,1,0 \
    PATH="/home/forge/.local/bin:${PATH}"

CMD ["forge", "--help"]