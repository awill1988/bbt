#!/usr/bin/env bash
# docker build script for bbt with multiple variants

set -e

# colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# defaults
IMAGE_NAME="${IMAGE_NAME:-bbt}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
ENABLE_GPU="${ENABLE_GPU:-false}"
RUST_VERSION="${RUST_VERSION:-nightly}"
CUDA_VERSION="${CUDA_VERSION:-12.4.0}"
PUSH="${PUSH:-false}"
PLATFORM="${PLATFORM:-linux/amd64}"

show_usage() {
    cat <<EOF
${GREEN}bbt docker build script${NC}

usage: $0 [options]

${YELLOW}options:${NC}
  -h, --help              show this help message
  -n, --name NAME         image name (default: bbt)
  -t, --tag TAG           image tag (default: latest)
  -g, --gpu               enable gpu support (default: false)
  -c, --cpu               cpu-only build (default)
  -p, --push              push to registry after build
  --platform PLATFORM     target platform (default: linux/amd64)
  --rust-version VERSION  rust version (default: nightly)
  --cuda-version VERSION  cuda version (default: 12.4.0)

${YELLOW}examples:${NC}
  # cpu-only build (default)
  $0 --cpu

  # gpu build
  $0 --gpu

  # custom tag
  $0 --tag v1.0.0 --cpu

  # multi-architecture build
  $0 --platform linux/amd64,linux/arm64 --cpu

  # build and push
  $0 --gpu --push --tag production

${YELLOW}presets:${NC}
  # development (cpu, fast builds)
  $0 --tag dev --cpu

  # production (optimized, cpu)
  $0 --tag prod --cpu

  # production with gpu
  $0 --tag prod-gpu --gpu

EOF
}

# parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -n|--name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -g|--gpu)
            ENABLE_GPU=true
            shift
            ;;
        -c|--cpu)
            ENABLE_GPU=false
            shift
            ;;
        -p|--push)
            PUSH=true
            shift
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --rust-version)
            RUST_VERSION="$2"
            shift 2
            ;;
        --cuda-version)
            CUDA_VERSION="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}unknown option: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

# determine build context
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo -e "${BLUE}=== bbt docker build ===${NC}"
echo
echo -e "${YELLOW}configuration:${NC}"
echo "  image:        ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  gpu enabled:  ${ENABLE_GPU}"
echo "  platform:     ${PLATFORM}"
echo "  rust version: ${RUST_VERSION}"
echo "  cuda version: ${CUDA_VERSION}"
echo "  push:         ${PUSH}"
echo

# build command
BUILD_CMD="docker build"

# use buildx for multi-platform builds
if [[ "$PLATFORM" == *","* ]]; then
    echo -e "${YELLOW}multi-platform build detected, using buildx${NC}"
    BUILD_CMD="docker buildx build --platform ${PLATFORM}"

    if [ "$PUSH" = true ]; then
        BUILD_CMD="$BUILD_CMD --push"
    else
        BUILD_CMD="$BUILD_CMD --load"
    fi
fi

# build arguments
BUILD_ARGS=(
    --build-arg "ENABLE_GPU=${ENABLE_GPU}"
    --build-arg "RUST_VERSION=${RUST_VERSION}"
    --build-arg "CUDA_VERSION=${CUDA_VERSION}"
    --tag "${IMAGE_NAME}:${IMAGE_TAG}"
)

# add cache mounts for faster rebuilds
BUILD_ARGS+=(
    --build-arg BUILDKIT_INLINE_CACHE=1
)

# build the image
echo -e "${GREEN}building docker image...${NC}"
cd "$PROJECT_ROOT"

$BUILD_CMD "${BUILD_ARGS[@]}" .

if [ $? -eq 0 ]; then
    echo
    echo -e "${GREEN}✓ build successful${NC}"
    echo
    echo -e "${YELLOW}image details:${NC}"
    docker images "${IMAGE_NAME}:${IMAGE_TAG}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    echo

    # push if requested (and not using buildx with --push)
    if [ "$PUSH" = true ] && [[ "$PLATFORM" != *","* ]]; then
        echo -e "${YELLOW}pushing to registry...${NC}"
        docker push "${IMAGE_NAME}:${IMAGE_TAG}"
        echo -e "${GREEN}✓ pushed successfully${NC}"
    fi

    echo -e "${YELLOW}run the image:${NC}"
    echo "  # basic run"
    echo "  docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} --help"
    echo
    echo "  # sync documents (set qdrant url as needed)"
    echo "  docker run --rm \\"
    echo "    -v \$(pwd)/data:/app/data \\"
    echo "    -v \$(pwd)/documents:/app/documents \\"
    echo "    -e QDRANT_URL=http://host.docker.internal:6334 \\"
    echo "    ${IMAGE_NAME}:${IMAGE_TAG} sync /app/documents"
    echo
    echo "  # query"
    echo "  docker run --rm \\"
    echo "    -e QDRANT_URL=http://host.docker.internal:6334 \\"
    echo "    ${IMAGE_NAME}:${IMAGE_TAG} query \"search text\""
    echo

    if [ "$ENABLE_GPU" = true ]; then
        echo "  # run with gpu (nvidia)"
        echo "  docker run --rm --gpus all ${IMAGE_NAME}:${IMAGE_TAG} --help"
        echo
    fi

    echo -e "${YELLOW}or use docker compose:${NC}"
    echo "  docker compose up -d qdrant otel-collector"
    echo
else
    echo
    echo -e "${RED}✗ build failed${NC}"
    exit 1
fi
