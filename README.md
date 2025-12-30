# bbt (big brain time)

Rust RAG system with hybrid search (vector + BM25), GPU acceleration, and real-time resource monitoring.

## Quick Start

```bash
# docker
docker compose up -d bbt-cpu qdrant

# native (macos)
brew install pdfium
cargo build --release --features coreml
./target/release/bbt sync ~/projects
./target/release/bbt query --query "search text"

# native (linux + nvidia)
cargo build --release --features cuda
```

## Commands

```bash
# sync files (scans git repos, honors .gitignore)
bbt sync ~/projects
bbt sync ./src --ext rs,toml,md
bbt sync ./docs --force --reset-state

# query
bbt query --query "search text"
bbt query --query "search text" --mode vector --top-k 10

# serve api
bbt serve --host 0.0.0.0 --port 8080
```

**Resource monitoring** (during sync):
```
⠋ cpu: 45.2% │ gpu: 78.5% │ mem: 3.2 GiB / 16.0 GiB │ vram: 1.8 GiB / 8.0 GiB
```

## Configuration

Key environment variables (create `.env`):

```bash
QDRANT_URL=http://localhost:6334
STATE_STORE_PATH=./data/state.db
CHUNK_SIZE=512
EMBEDDING_MODEL_REPO=BAAI/bge-small-en-v1.5
EMBEDDING_BATCH_SIZE=32
FORCE_CPU=0                    # 1 to disable gpu
LOG_LEVEL=info
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

## GPU Memory Management

Large files get exclusive GPU access to prevent OOM:

```bash
bbt sync ~/projects \
  --large-file-threshold-kib 20 \  # files > 20 KiB get exclusive access
  --max-pool-bytes-kib 128         # max concurrent bytes in flight
```

## Architecture

```
bbt/       - cli (sync, query, serve)
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
docker compose up -d jaeger        # tracing:   http://localhost:16686
docker compose up -d langfuse-web  # observability: http://localhost:3000
```

## License

MIT
