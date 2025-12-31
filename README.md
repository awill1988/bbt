# bbt (big brain time)

Rust RAG system with hybrid search (vector + BM25), GPU acceleration, and real-time resource monitoring.

## Quick Start

```bash
# docker deps
docker compose up -d qdrant otel-collector

# docker cli (cpu, scans ./documents)
docker compose --profile cpu run --rm bbt-cpu sync /app/documents
docker compose --profile cpu run --rm bbt-cpu query "search text"

# native (macos)
brew install pdfium
cargo build --release --features coreml

# native (linux + nvidia)
cargo build --release --features cuda

# run
./target/release/bbt sync ~/projects
./target/release/bbt query "search text"
```

## Commands

```bash
# sync files (scans git repos, honors .gitignore)
bbt sync ~/projects
bbt sync ./src --ext rs,toml,md
bbt sync ./docs --force --reset-state

# query
bbt query "search text"
bbt query "search text" --mode vector --top-k 10
```

**Resource monitoring** (during sync):
```
⠋ cpu: 45.2% │ gpu: 78.5% │ mem: 3.2 GiB / 16.0 GiB │ vram: 1.8 GiB / 8.0 GiB
```

## Configuration

Defaults live in `backbone/src/config/defaults.rs`. Override only what you need:

```bash
LOG_LEVEL=info
QDRANT_URL=http://localhost:6334
EMBEDDING_MODEL_REPO=jinaai/jina-embeddings-v2-base-code
EMBEDDING_MODEL_FILE=onnx/model.onnx
EMBEDDING_BATCH_SIZE=32
FORCE_CPU=1
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

Qdrant defaults to the gRPC port (6334).
Set `FORCE_CPU=1` to disable gpu acceleration.
Data defaults to `./data` (state db: `./data/state.db`) and models to `./.cache/models`. Override with `DATA_DIR`, `STATE_STORE_PATH`, or `MODEL_CACHE_DIR` if needed.
Tracing is enabled when an OTLP endpoint is set. Disable with `ENABLE_TRACING=false`.

## GPU Memory Management

Large files get exclusive GPU access to prevent OOM:

```bash
bbt sync ~/projects \
  --large-file-threshold-kib 20 \
  --max-pool-bytes-kib 128
```

Files over 20 KiB get exclusive GPU access; max in-flight bytes is 128 KiB.

## Architecture

```
bbt/       - cli (sync, query, gen)
backbone/  - core library (embedding, chunking, storage, tracing)
```

## Heap Profiling

Build and run with profiling enabled:

```bash
cargo build --release --features heap-profiling
cargo run --release --features heap-profiling -- sync ~/projects
```

Snapshots written every 50 files to `dhat-heap-<count>-files-<timestamp>.json`.

View profiles:
```bash
./scripts/view-dhat.sh dhat-heap.json
# or open https://nnethercote.github.io/dh_view/dh_view.html
```

## Services

```bash
docker compose up -d qdrant        # vector db: http://localhost:6333/dashboard
docker compose up -d otel-collector # otlp grpc: http://localhost:4317
docker compose up -d jaeger        # tracing:   http://localhost:16686
docker compose up -d langfuse-web  # observability: http://localhost:3000
```

## License

MIT
