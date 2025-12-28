use super::{defaults, BbtConfig};
use crate::config::feature_flags::FeatureFlags;
use crate::error::{BbtError, Result};
use std::env;
use std::path::PathBuf;
use std::str::FromStr;

/// Parse configuration from environment variables
pub fn from_env() -> Result<BbtConfig> {
    let defaults = defaults::default_config();

    let config = BbtConfig {
        // General
        log_level: get_env_or("LOG_LEVEL", defaults.log_level),

        // Paths
        data_dir: get_env_path_or("BBT_DATA_DIR", defaults.data_dir),
        model_cache_dir: get_env_path_or("BBT_MODEL_CACHE_DIR", defaults.model_cache_dir),
        vector_store_path: get_env_path_or("BBT_VECTOR_STORE_PATH", defaults.vector_store_path),
        bm25_index_path: get_env_path_or("BBT_BM25_INDEX_PATH", defaults.bm25_index_path),
        state_store_path: get_env_path_or("BBT_STATE_STORE_PATH", defaults.state_store_path),

        // Document processing
        chunk_size: get_env_parse_or("BBT_CHUNK_SIZE", defaults.chunk_size)?,
        chunk_overlap: get_env_parse_or("BBT_CHUNK_OVERLAP", defaults.chunk_overlap)?,
        max_file_size_bytes: get_env_parse_or("BBT_MAX_FILE_SIZE_BYTES", defaults.max_file_size_bytes)?,
        enable_ocr: get_env_bool_or("BBT_ENABLE_OCR", defaults.enable_ocr)?,
        enable_section_detection: get_env_bool_or(
            "BBT_ENABLE_SECTION_DETECTION",
            defaults.enable_section_detection,
        )?,
        enable_header_footer_removal: get_env_bool_or(
            "BBT_ENABLE_HEADER_FOOTER_REMOVAL",
            defaults.enable_header_footer_removal,
        )?,

        // Embedding
        embedding_model_repo: get_env_or("BBT_EMBEDDING_MODEL_REPO", defaults.embedding_model_repo),
        embedding_model_file: get_env_or("BBT_EMBEDDING_MODEL_FILE", defaults.embedding_model_file),
        embedding_batch_size: get_env_parse_or(
            "BBT_EMBEDDING_BATCH_SIZE",
            defaults.embedding_batch_size,
        )?,
        embedding_dims: get_env_parse_or("BBT_EMBEDDING_DIMS", defaults.embedding_dims)?,

        // Retrieval
        retrieval_mode: get_env_parse_or("BBT_RETRIEVAL_MODE", defaults.retrieval_mode)?,
        top_k: get_env_parse_or("BBT_TOP_K", defaults.top_k)?,
        min_score: get_env_parse_or("BBT_MIN_SCORE", defaults.min_score)?,
        vector_weight: get_env_parse_or("BBT_VECTOR_WEIGHT", defaults.vector_weight)?,
        bm25_weight: get_env_parse_or("BBT_BM25_WEIGHT", defaults.bm25_weight)?,
        enable_rerank: get_env_bool_or("BBT_ENABLE_RERANK", defaults.enable_rerank)?,
        rerank_model_repo: get_env_or("BBT_RERANK_MODEL_REPO", defaults.rerank_model_repo),
        rerank_model_file: get_env_or("BBT_RERANK_MODEL_FILE", defaults.rerank_model_file),
        rerank_top_k: get_env_parse_or("BBT_RERANK_TOP_K", defaults.rerank_top_k)?,

        // BM25
        bm25_k1: get_env_parse_or("BBT_BM25_K1", defaults.bm25_k1)?,
        bm25_b: get_env_parse_or("BBT_BM25_B", defaults.bm25_b)?,

        // Vector store
        vector_store_type: get_env_parse_or(
            "BBT_VECTOR_STORE_TYPE",
            defaults.vector_store_type,
        )?,
        qdrant_url: get_env_or_fallback(
            "BBT_QDRANT_URL",
            "BBT_VECTOR_STORE_URL",
            defaults.qdrant_url,
        ),
        qdrant_api_key: env::var("BBT_QDRANT_API_KEY").ok(),
        chromadb_url: get_env_or("BBT_CHROMADB_URL", defaults.chromadb_url),
        chromadb_collection: get_env_or("BBT_CHROMADB_COLLECTION", defaults.chromadb_collection),

        // Caching
        cache_max_age_secs: get_env_parse_or(
            "BBT_CACHE_MAX_AGE_SECS",
            defaults.cache_max_age_secs,
        )?,
        cache_max_entries: get_env_parse_or("BBT_CACHE_MAX_ENTRIES", defaults.cache_max_entries)?,

        // Server
        server_host: get_env_or("BBT_SERVER_HOST", defaults.server_host),
        server_port: get_env_parse_or("BBT_SERVER_PORT", defaults.server_port)?,

        // OpenTelemetry
        enable_tracing: get_env_bool_or("BBT_ENABLE_TRACING", defaults.enable_tracing)?,
        otel_endpoint: env::var("BBT_OTEL_ENDPOINT").ok(),

        // Feature flags
        features: FeatureFlags {
            enable_pdf_ocr: get_env_bool_or(
                "BBT_FEATURE_PDF_OCR",
                defaults.features.enable_pdf_ocr,
            )?,
            enable_reranking: get_env_bool_or(
                "BBT_FEATURE_RERANKING",
                defaults.features.enable_reranking,
            )?,
            enable_section_chunking: get_env_bool_or(
                "BBT_FEATURE_SECTION_CHUNKING",
                defaults.features.enable_section_chunking,
            )?,
            enable_citation_formatting: get_env_bool_or(
                "BBT_FEATURE_CITATION_FORMATTING",
                defaults.features.enable_citation_formatting,
            )?,
            enable_context_compression: get_env_bool_or(
                "BBT_FEATURE_CONTEXT_COMPRESSION",
                defaults.features.enable_context_compression,
            )?,
            enable_extractive_summary: get_env_bool_or(
                "BBT_FEATURE_EXTRACTIVE_SUMMARY",
                defaults.features.enable_extractive_summary,
            )?,
        },
    };

    config.validate()?;
    Ok(config)
}

/// Get environment variable as string or use default
fn get_env_or(key: &str, default: String) -> String {
    env::var(key).unwrap_or(default)
}

/// Get environment variable as string with fallback
fn get_env_or_fallback(primary: &str, fallback: &str, default: String) -> String {
    env::var(primary).unwrap_or_else(|_| env::var(fallback).unwrap_or(default))
}

/// Get environment variable as PathBuf or use default
fn get_env_path_or(key: &str, default: PathBuf) -> PathBuf {
    env::var(key).map(PathBuf::from).unwrap_or(default)
}

/// Parse environment variable or use default
fn get_env_parse_or<T>(key: &str, default: T) -> Result<T>
where
    T: FromStr + Clone,
    T::Err: std::fmt::Display,
{
    match env::var(key) {
        Ok(val) => val.parse::<T>().map_err(|e| {
            BbtError::Schema(format!("failed to parse env var {}: {}", key, e))
        }),
        Err(_) => Ok(default),
    }
}

/// Parse boolean environment variable or use default
fn get_env_bool_or(key: &str, default: bool) -> Result<bool> {
    match env::var(key) {
        Ok(val) => {
            let val_lower = val.to_lowercase();
            match val_lower.as_str() {
                "true" | "1" | "yes" | "on" => Ok(true),
                "false" | "0" | "no" | "off" => Ok(false),
                _ => Err(BbtError::Schema(format!(
                    "invalid boolean value for {}: {}",
                    key, val
                ))),
            }
        }
        Err(_) => Ok(default),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_env_defaults() {
        // Clear any existing env vars
        std::env::remove_var("BBT_CHUNK_SIZE");
        std::env::remove_var("LOG_LEVEL");

        let config = from_env().unwrap();
        assert_eq!(config.chunk_size, 512);
        assert_eq!(config.log_level, "info");
    }

    #[test]
    fn test_env_parsing() {
        std::env::set_var("BBT_CHUNK_SIZE", "1024");
        std::env::set_var("BBT_ENABLE_OCR", "true");
        std::env::set_var("BBT_RETRIEVAL_MODE", "vector");

        let config = from_env().unwrap();
        assert_eq!(config.chunk_size, 1024);
        assert_eq!(config.enable_ocr, true);
        assert_eq!(config.retrieval_mode, RetrievalMode::Vector);

        // Cleanup
        std::env::remove_var("BBT_CHUNK_SIZE");
        std::env::remove_var("BBT_ENABLE_OCR");
        std::env::remove_var("BBT_RETRIEVAL_MODE");
    }
}
