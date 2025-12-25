#!/usr/bin/env python3
"""
export nemotron embedding model to onnx format

requires:
    pip install optimum[exporters] onnxruntime transformers torch
"""

import argparse
from pathlib import Path
from optimum.exporters.onnx import main_export


def export_nemotron_to_onnx(
    model_id: str = "nvidia/llama-embed-nemotron-8b",
    output_dir: str = "./.cache/models/nvidia_llama-embed-nemotron-8b/onnx",
):
    """
    export nvidia nemotron embedding model to onnx format

    args:
        model_id: huggingface model id
        output_dir: output directory for onnx model
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"exporting {model_id} to onnx format...")
    print(f"output directory: {output_path}")

    # export using optimum
    main_export(
        model_name_or_path=model_id,
        output=str(output_path),
        task="feature-extraction",  # embedding task
        opset=14,  # onnx opset version
        device="cpu",  # export on cpu, can run on gpu later
        fp16=False,  # use fp32 for better compatibility
    )

    print(f"export complete! model saved to {output_path}")
    print("\nto use this model, set in your .env:")
    print(f'BBT_EMBEDDING_MODEL_REPO=nvidia/llama-embed-nemotron-8b')
    print(f'BBT_EMBEDDING_MODEL_FILE=onnx/model.onnx')
    print(f'BBT_EMBEDDING_DIMS=4096')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="export nemotron to onnx")
    parser.add_argument(
        "--model-id",
        type=str,
        default="nvidia/llama-embed-nemotron-8b",
        help="huggingface model id",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./.cache/models/nvidia_llama-embed-nemotron-8b/onnx",
        help="output directory for onnx model",
    )

    args = parser.parse_args()
    export_nemotron_to_onnx(args.model_id, args.output_dir)
