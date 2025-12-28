use super::{BbtConfig, RetrievalMode, VectorStoreType};
use crate::config::feature_flags::FeatureFlags;
use std::path::PathBuf;

/// Default configuration values
pub fn default_config() -> BbtConfig {
    BbtConfig {
        // General
        log_level: "info".to_string(),

        // Paths
        data_dir: PathBuf::from("./data"),
        model_cache_dir: PathBuf::from("./.cache/models"),
        vector_store_path: PathBuf::from("./data/vector_store"),
        bm25_index_path: PathBuf::from("./data/bm25_index.bin"),
        state_store_path: PathBuf::from("./data/state.db"),

        // Document processing
        chunk_size: 512,
        chunk_overlap: 128,
        max_file_size_bytes: 10_485_760, // 10 MiB
        enable_ocr: false,
        enable_section_detection: true,
        enable_header_footer_removal: true,

        // Embedding (jina-embeddings-v2-base-code: 768 dims, 8192 token context)
        embedding_model_repo: "jinaai/jina-embeddings-v2-base-code".to_string(),
        embedding_model_file: "onnx/model.onnx".to_string(),
        embedding_batch_size: 32,
        embedding_dims: 768,

        // Retrieval
        retrieval_mode: RetrievalMode::Hybrid,
        top_k: 5,
        min_score: 0.5,
        vector_weight: 0.5,
        bm25_weight: 0.5,
        enable_rerank: false,
        rerank_model_repo: "cross-encoder/ms-marco-MiniLM-L-6-v2".to_string(),
        rerank_model_file: "onnx/model.onnx".to_string(),
        rerank_top_k: 5,

        // BM25
        bm25_k1: 1.5,
        bm25_b: 0.75,

        // Vector store
        vector_store_type: VectorStoreType::Qdrant,
        qdrant_url: "http://localhost:6334".to_string(),
        qdrant_api_key: None,
        chromadb_url: "http://localhost:8000".to_string(),
        chromadb_collection: "bbt".to_string(),

        // Caching
        cache_max_age_secs: 86400, // 24 hours
        cache_max_entries: 10000,

        // Server
        server_host: "0.0.0.0".to_string(),
        server_port: 8080,

        // OpenTelemetry
        enable_tracing: true,
        otel_endpoint: None,

        // Feature flags
        features: FeatureFlags::default(),
    }
}
