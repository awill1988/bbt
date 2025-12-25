# scripts

utility scripts for model management and deployment.

## export nemotron embedding model to onnx

the nvidia nemotron embedding model needs to be exported to onnx format before use.

### prerequisites

```bash
pip install optimum[exporters] onnxruntime transformers torch
```

### export the model

```bash
python scripts/export_nemotron_onnx.py
```

this will:
1. download the nvidia/llama-embed-nemotron-8b model from huggingface
2. export it to onnx format using optimum
3. save to `.cache/models/nvidia_llama-embed-nemotron-8b/onnx/`

the export takes ~5-10 minutes and requires ~32gb disk space for the model weights.

### use the exported model

create or edit `.env`:

```bash
BBT_EMBEDDING_MODEL_REPO=nvidia/llama-embed-nemotron-8b
BBT_EMBEDDING_MODEL_FILE=onnx/model.onnx
BBT_EMBEDDING_DIMS=4096
BBT_EMBEDDING_BATCH_SIZE=16
```

then run bbt as normal:

```bash
bbt sync ./data
```

## notes

- the nemotron model produces 4096-dimensional embeddings (vs 384 for bge-small)
- larger batch sizes may cause oom on systems with limited ram
- first inference run may be slow as onnx runtime optimizes the graph
