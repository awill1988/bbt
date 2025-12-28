# syntax=docker/dockerfile:1

# =============================================================================
# Build arguments
# =============================================================================
ARG RUST_VERSION=nightly
ARG CUDA_VERSION=12.4.0
ARG UBUNTU_VERSION=22.04
ARG TARGETPLATFORM=linux/amd64
ARG BUILDPLATFORM=linux/amd64

# =============================================================================
# Base images
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder-gpu
FROM rustlang/rust:${RUST_VERSION}-slim AS builder-cpu

# =============================================================================
# Builder stage (GPU variant)
# =============================================================================
FROM builder-gpu AS builder-gpu-deps

# install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    curl \
    libssl-dev \
    libclang-dev \
    && rm -rf /var/lib/apt/lists/*

# install rust toolchain
ARG RUST_VERSION=nightly
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain ${RUST_VERSION}
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build

# =============================================================================
# Builder stage (CPU variant)
# =============================================================================
FROM builder-cpu AS builder-cpu-deps

# install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libclang-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# =============================================================================
# Dependency caching stage
# =============================================================================
ARG ENABLE_GPU=false
FROM builder-${ENABLE_GPU:+gpu}${ENABLE_GPU:-cpu}-deps AS deps

# copy only dependency files for layer caching
COPY Cargo.toml Cargo.lock ./
COPY backbone/Cargo.toml backbone/Cargo.toml
COPY bbt/Cargo.toml bbt/Cargo.toml

# create dummy source files to build dependencies
RUN mkdir -p backbone/src bbt/src && \
    echo "fn main() {}" > bbt/src/main.rs && \
    echo "pub fn lib() {}" > backbone/src/lib.rs

# build dependencies only (cached layer, allow it to fail on dummy code)
RUN cargo build --release || true
RUN rm -rf target/release/.fingerprint/bbt-* target/release/.fingerprint/backbone-* \
    target/release/bbt* target/release/backbone* target/release/deps/bbt-* target/release/deps/backbone-*

# =============================================================================
# Build stage
# =============================================================================
FROM deps AS builder

# reuse gpu flag for feature selection
ARG ENABLE_GPU=false

# remove dummy files
RUN rm -rf backbone/src bbt/src

# copy actual source code
COPY backbone ./backbone
COPY bbt ./bbt

# build the actual binary with optimizations
ARG RUSTFLAGS="-C target-cpu=native -C opt-level=3"
ENV RUSTFLAGS=${RUSTFLAGS}

RUN if [ "$ENABLE_GPU" = "true" ]; then \
        cargo build --release --locked --bin bbt --features cuda; \
    else \
        cargo build --release --locked --bin bbt; \
    fi

# strip debug symbols to reduce binary size
RUN strip target/release/bbt

# =============================================================================
# Runtime base (GPU variant)
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime-gpu-base

# install runtime dependencies for gpu
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libssl3 \
    ca-certificates \
    curl \
    # pdfium dependencies
    libfreetype6 \
    libfontconfig1 \
    libjpeg-turbo8 \
    libpng16-16 \
    # tesseract (optional for ocr)
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    # pdfium library
    wget \
    && rm -rf /var/lib/apt/lists/*

# install pdfium library (architecture-specific)
ARG TARGETARCH
RUN case "${TARGETARCH}" in \
        amd64) PDFIUM_ARCH="x64" ;; \
        arm64) PDFIUM_ARCH="arm64" ;; \
        *) echo "unsupported architecture: ${TARGETARCH}" && exit 1 ;; \
    esac && \
    wget -O /tmp/pdfium.tgz https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-linux-${PDFIUM_ARCH}.tgz && \
    tar -xzf /tmp/pdfium.tgz -C /usr/local && \
    rm /tmp/pdfium.tgz && \
    ldconfig

# =============================================================================
# Runtime base (CPU variant)
# =============================================================================
FROM debian:bookworm-slim AS runtime-cpu-base

# install runtime dependencies for cpu
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libssl3 \
    ca-certificates \
    curl \
    # pdfium dependencies
    libfreetype6 \
    libfontconfig1 \
    libjpeg62-turbo \
    libpng16-16 \
    # tesseract (optional for ocr)
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    # pdfium library
    wget \
    && rm -rf /var/lib/apt/lists/*

# install pdfium library (architecture-specific)
ARG TARGETARCH
RUN case "${TARGETARCH}" in \
        amd64) PDFIUM_ARCH="x64" ;; \
        arm64) PDFIUM_ARCH="arm64" ;; \
        *) echo "unsupported architecture: ${TARGETARCH}" && exit 1 ;; \
    esac && \
    wget -O /tmp/pdfium.tgz https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-linux-${PDFIUM_ARCH}.tgz && \
    tar -xzf /tmp/pdfium.tgz -C /usr/local && \
    rm /tmp/pdfium.tgz && \
    ldconfig

# =============================================================================
# Final runtime stage
# =============================================================================
ARG ENABLE_GPU=false
FROM runtime-${ENABLE_GPU:+gpu}${ENABLE_GPU:-cpu}-base AS runtime

WORKDIR /app

# copy binary from builder
COPY --from=builder /build/target/release/bbt /usr/local/bin/bbt

# create necessary directories with proper permissions
RUN mkdir -p /app/.cache /app/data /app/documents /app/models && \
    # create non-root user
    groupadd -g 1000 bbt && \
    useradd -u 1000 -g bbt -s /bin/bash -m bbt && \
    # set ownership
    chown -R bbt:bbt /app

# set environment variables
ENV LOG_LEVEL=info \
    BBT_DATA_DIR=/app/data \
    BBT_MODEL_CACHE_DIR=/app/models \
    HF_HOME=/app/.cache \
    # pdfium library path
    LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH} \
    # disable gpu by default (can be overridden)
    BBT_FORCE_CPU=${ENABLE_GPU:+0}${ENABLE_GPU:-1}

# expose api port
EXPOSE 8080

# health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# switch to non-root user
USER bbt:bbt

ENTRYPOINT ["bbt"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]

# =============================================================================
# Metadata
# =============================================================================
LABEL org.opencontainers.image.title="bbt" \
      org.opencontainers.image.description="big brain time - RAG system with hybrid search" \
      org.opencontainers.image.vendor="bbt" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/awill1988/bookmarks"
