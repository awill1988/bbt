#!/usr/bin/env bash
set -euo pipefail

CUDA_IMAGE="${CUDA_IMAGE:-nvidia/cuda:12.4.0-base-ubuntu22.04}"
BBT_IMAGE="${BBT_IMAGE:-bbt:latest}"
RUN_BBT=false
RUN_COMPOSE=false

show_usage() {
    cat <<EOF
bbt gpu check (docker)

usage: $0 [options]

options:
  --bbt      run bbt image with --gpus all
  --compose  run docker compose profile nvidia for bbt
  -h, --help show this help message

env:
  CUDA_IMAGE  nvidia/cuda image (default: nvidia/cuda:12.4.0-base-ubuntu22.04)
  BBT_IMAGE   bbt image (default: bbt:latest)
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --bbt)
            RUN_BBT=true
            shift
            ;;
        --compose)
            RUN_COMPOSE=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "error: unknown option: $1" >&2
            show_usage >&2
            exit 1
            ;;
    esac
done

if ! command -v docker >/dev/null 2>&1; then
    echo "error: docker not found in path" >&2
    exit 1
fi

echo "checking gpu access with docker..."
echo "image: ${CUDA_IMAGE}"

if docker run --rm --gpus all "${CUDA_IMAGE}" nvidia-smi >/dev/null; then
    echo "gpu available in docker"
else
    echo "gpu not available in docker" >&2
    echo "hint: install nvidia-container-toolkit and restart docker" >&2
    exit 1
fi

if [ "${RUN_BBT}" = true ]; then
    echo "checking bbt image gpu startup..."
    docker run --rm --gpus all "${BBT_IMAGE}" --help >/dev/null
    echo "bbt image started with gpu"
fi

if [ "${RUN_COMPOSE}" = true ]; then
    echo "checking docker compose nvidia profile..."
    docker compose --profile nvidia run --rm bbt --help >/dev/null
    echo "docker compose nvidia profile ok"
fi
