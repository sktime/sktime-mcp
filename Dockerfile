# =============================================================================
# Dockerfile for sktime-mcp
#
# Multi-stage build:
#   1. Builder  – installs build tools and Python dependencies
#   2. Runtime  – slim image with only the installed packages
#
# Usage:
#   docker build -t sktime-mcp .
#   docker run -i sktime-mcp                  # stdio transport (default)
#   docker run -i -e SKTIME_MCP_LOG_LEVEL=DEBUG sktime-mcp
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# Avoid interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# Install minimal build dependencies required by some scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only dependency-related files first to maximise Docker layer caching
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

# Install the project in production mode (no dev/test extras)
RUN pip install --no-cache-dir --prefix=/install .

# ---------------------------------------------------------------------------
# Stage 2: Runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Runtime library needed by numpy/scipy linked against OpenBLAS
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from the builder stage
COPY --from=builder /install /usr/local

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash mcp
USER mcp
WORKDIR /home/mcp

# Environment defaults (overridable at runtime)
ENV SKTIME_MCP_LOG_LEVEL=WARNING
ENV PYTHONUNBUFFERED=1

# Health-check: verify the package is importable
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import sktime_mcp; print('ok')" || exit 1

# MCP servers communicate over stdio, so the entrypoint is the CLI command
ENTRYPOINT ["sktime-mcp"]
