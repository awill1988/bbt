# bbt (big brain time)

Rust-based RAG system with hybrid search (vector + BM25), PDF processing, and GPU acceleration.

## Quick Start

### Docker (Recommended)

```bash
# start services
docker compose up -d bbt-cpu qdrant otel-collector

# check health
curl http://localhost:8080/health

# ingest documents
curl -X POST http://localhost:8080/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"path": "/app/documents", "recursive": true}'

# query
curl -X POST http://localhost:8080/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "your search", "mode": "hybrid", "top_k": 5}'
```

**GPU support (NVIDIA)**:
```bash
docker compose --profile nvidia up -d bbt
```

### Native Build

**Prerequisites**:
- Rust 1.83+ (`rustup install stable`)
- pdfium (`brew install pdfium` or `apt-get install libpdfium-dev`)
- tesseract (optional, for OCR)

**Build**:
```bash
cargo build --release --locked
./target/release/bbt --help
```

**Linux with NVIDIA GPU**:
```bash
cargo build --release --locked --features cuda
```

**macOS with CoreML GPU** (4-6x faster than CPU):
```bash
# coreml only works natively, not in docker
brew install pdfium tesseract
cargo build --release --locked --features coreml

# verify coreml detection
LOG_LEVEL=info ./target/release/bbt serve
# should show: provider=coreml

# force cpu if needed
export BBT_FORCE_CPU=1
```

## Commands

### Ingest / Sync

**Sync source code from git repositories:**
```bash
# default: scans git repos only, honors .gitignore, indexes code files
bbt sync ~/projects

# custom extensions
bbt sync ./src --ext rs,toml,md

# include all files (not just code)
bbt sync ./docs --ext docs

# disable git filtering
bbt sync ./all --git-only false --honor-gitignore false

# force reprocessing and reset state
bbt sync ./docs --force
bbt sync ./docs --reset-state
```

**Extension presets:**
- `code` - js, jsx, ts, tsx, rust, go, python + config files (default)
- `docs` - md, txt, rst, adoc
- custom - specify exact extensions: `--ext rs,toml,py`

### Query
```bash
bbt query --query "search text"
bbt query --query "search text" --mode vector --top-k 10
bbt query --query "search text" --mode bm25 --show-scores
bbt query --query "search text" --citation-mode hash
BBT_CITATION_SECRET=local_dev_secret bbt query --query "search text" --citation-mode hash
```
note: bm25/hybrid modes currently fall back to vector retrieval.

### Serve
```bash
bbt serve --host 0.0.0.0 --port 8080
```

## Configuration

Create `.env`:
```bash
# data
BBT_DATA_DIR=./data
BBT_QDRANT_URL=http://localhost:6334
BBT_STATE_STORE_PATH=./data/state.db
# use .db/.sqlite for sqlite, .json for legacy json state

# chunking
BBT_CHUNK_SIZE=512
BBT_CHUNK_OVERLAP=128

# embedding
BBT_EMBEDDING_MODEL_REPO=BAAI/bge-small-en-v1.5
BBT_EMBEDDING_BATCH_SIZE=32
BBT_EMBEDDING_WORKERS=8
BBT_EMBEDDING_QUEUE_SIZE=32

# retrieval
BBT_RETRIEVAL_MODE=hybrid
BBT_TOP_K=20
BBT_VECTOR_WEIGHT=0.7
BBT_BM25_WEIGHT=0.3

# gpu
BBT_FORCE_CPU=0  # 0=auto-detect gpu, 1=force cpu

# tracing
BBT_ENABLE_TRACING=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# logging
LOG_LEVEL=info

# sync control
BBT_FORCE_SYNC=false
BBT_RESET_STATE=false
```

## API Endpoints

```bash
# health
GET /health

# ingest
POST /api/v1/ingest
{
  "path": "./documents",
  "recursive": true,
  "chunk_size": 512,
  "enable_ocr": false
}

# query
POST /api/v1/query
{
  "query": "search text",
  "mode": "hybrid",
  "top_k": 5,
  "show_scores": true
}

# status
GET /api/v1/status

# clear cache
DELETE /api/v1/cache
```

## Docker Build

```bash
# cpu build
docker build --build-arg ENABLE_GPU=false -t bbt:cpu .

# gpu build (nvidia)
docker build --build-arg ENABLE_GPU=true -t bbt:gpu .

# multi-arch (linux/amd64,linux/arm64)
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --build-arg ENABLE_GPU=false \
  -t bbt:latest .

# or use script
./scripts/build-docker.sh --cpu --tag v1.0.0
./scripts/build-docker.sh --gpu --tag v1.0.0-gpu
```

## Architecture

```
bbt/           - cli binary (ingest, query, serve, gen)
backbone/      - core library
  ├── config/      - configuration
  ├── document/    - pdf/text processing, chunking
  ├── embedding/   - onnx embedding (coreml/cuda/cpu)
  ├── storage/     - state tracking, metadata
  ├── agent/       - llm agents
  ├── dag/         - task orchestration
  └── tracing/     - opentelemetry
```

## GPU Support

**Auto-detected** for embedding generation:
- **macOS (M1/M2/M3)**: CoreML backend (native build only, not docker)
- **Linux + NVIDIA**: CUDA backend (requires nvidia-docker)
- **Other**: CPU fallback

**Performance (embedding generation)**:
| Platform | Backend | Throughput | Speedup |
|----------|---------|------------|---------|
| M1 Pro | CoreML | ~800 chunks/sec | 5x |
| M2 Max | CoreML | ~1200 chunks/sec | 5x |
| M3 Max | CoreML | ~1500 chunks/sec | 5x |
| RTX 3090 | CUDA | ~2000 chunks/sec | 8x |

**Note**: Docker on macOS runs Linux VM without CoreML. Use native build for GPU on Mac.

## Services

**Qdrant** (vector database):
```bash
docker run -p 6333:6333 -p 6334:6334 -v ./qdrant:/qdrant/storage qdrant/qdrant
# ui: http://localhost:6333/dashboard
```
note: bbt connects to qdrant over grpc on port 6334.

**Langfuse** (observability):
```bash
docker compose up -d langfuse-web
# ui: http://localhost:3000
# login: dev@example.com / local_dev_password
```

## Development

```bash
# test
cargo test --workspace

# run
cargo run --release -- serve

# format
cargo fmt

# lint
cargo clippy -- -D warnings

# with tracing
LOG_LEVEL=debug cargo run -- ingest ./documents
```

## Troubleshooting

**pdfium not found**:
```bash
brew install pdfium  # macos
apt-get install libpdfium-dev  # linux
```

**qdrant connection refused**:
```bash
docker compose ps qdrant
docker compose restart qdrant
curl http://localhost:6333/collections
```

**slow embeddings**:
```bash
# enable gpu (unset force_cpu)
unset BBT_FORCE_CPU

# increase batch size
export BBT_EMBEDDING_BATCH_SIZE=64
```

**coreml not detected (macos)**:
```bash
# verify coreml support
LOG_LEVEL=debug ./target/release/bbt serve 2>&1 | grep -i coreml

# check not running in docker
uname -s  # should be "Darwin", not "Linux"
```

## License

MIT
