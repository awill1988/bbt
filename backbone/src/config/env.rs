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
        log_level: get_env_or(&["LOG_LEVEL"], defaults.log_level),

        // Paths
        data_dir: get_env_path_or(&["DATA_DIR"], defaults.data_dir),
        model_cache_dir: get_env_path_or(&["MODEL_CACHE_DIR"], defaults.model_cache_dir),
        vector_store_path: get_env_path_or(&["VECTOR_STORE_PATH"], defaults.vector_store_path),
        bm25_index_path: get_env_path_or(&["BM25_INDEX_PATH"], defaults.bm25_index_path),
        state_store_path: get_env_path_or(&["STATE_STORE_PATH"], defaults.state_store_path),

        // Document processing
        chunk_size: get_env_parse_or(&["CHUNK_SIZE"], defaults.chunk_size)?,
        chunk_overlap: get_env_parse_or(
            &["CHUNK_OVERLAP"],
            defaults.chunk_overlap,
        )?,
        max_file_size_bytes: get_env_parse_or(
            &["MAX_FILE_SIZE_BYTES"],
            defaults.max_file_size_bytes,
        )?,
        enable_ocr: get_env_bool_or(&["ENABLE_OCR"], defaults.enable_ocr)?,
        enable_section_detection: get_env_bool_or(
            &["ENABLE_SECTION_DETECTION"],
            defaults.enable_section_detection,
        )?,
        enable_header_footer_removal: get_env_bool_or(
            &["ENABLE_HEADER_FOOTER_REMOVAL"],
            defaults.enable_header_footer_removal,
        )?,

        // Embedding
        embedding_model_repo: get_env_or(&["EMBEDDING_MODEL_REPO"], defaults.embedding_model_repo),
        embedding_model_file: get_env_or(&["EMBEDDING_MODEL_FILE"], defaults.embedding_model_file),
        embedding_batch_size: get_env_parse_or(
            &["EMBEDDING_BATCH_SIZE"],
            defaults.embedding_batch_size,
        )?,
        embedding_workers: get_env_parse_or(
            &["EMBEDDING_WORKERS"],
            defaults.embedding_workers,
        )?,
        embedding_queue_size: get_env_parse_or(
            &["EMBEDDING_QUEUE_SIZE"],
            defaults.embedding_queue_size,
        )?,
        embedding_dims: get_env_parse_or(
            &["EMBEDDING_DIMS"],
            defaults.embedding_dims,
        )?,
        embedding_max_seq_len: get_env_parse_or(
            &["EMBEDDING_MAX_SEQ_LEN"],
            defaults.embedding_max_seq_len,
        )?,

        // Retrieval
        retrieval_mode: get_env_parse_or(
            &["RETRIEVAL_MODE"],
            defaults.retrieval_mode,
        )?,
        top_k: get_env_parse_or(&["TOP_K"], defaults.top_k)?,
        min_score: get_env_parse_or(&["MIN_SCORE"], defaults.min_score)?,
        vector_weight: get_env_parse_or(
            &["VECTOR_WEIGHT"],
            defaults.vector_weight,
        )?,
        bm25_weight: get_env_parse_or(&["BM25_WEIGHT"], defaults.bm25_weight)?,
        adaptive_scoring: get_env_bool_or(&["ADAPTIVE_SCORING"], defaults.adaptive_scoring)?,
        adaptive_threshold: get_env_parse_or(&["ADAPTIVE_THRESHOLD"], defaults.adaptive_threshold)?,
        enable_rerank: get_env_bool_or(&["ENABLE_RERANK"], defaults.enable_rerank)?,
        rerank_model_repo: get_env_or(
            &["RERANK_MODEL_REPO"],
            defaults.rerank_model_repo,
        ),
        rerank_model_file: get_env_or(
            &["RERANK_MODEL_FILE"],
            defaults.rerank_model_file,
        ),
        rerank_top_k: get_env_parse_or(
            &["RERANK_TOP_K"],
            defaults.rerank_top_k,
        )?,

        // BM25
        bm25_k1: get_env_parse_or(&["BM25_K1"], defaults.bm25_k1)?,
        bm25_b: get_env_parse_or(&["BM25_B"], defaults.bm25_b)?,

        // Vector store
        vector_store_type: get_env_parse_or(
            &["VECTOR_STORE_TYPE"],
            defaults.vector_store_type,
        )?,
        qdrant_url: get_env_or(
            &[
                "QDRANT_URL",
                "VECTOR_STORE_URL",
            ],
            defaults.qdrant_url,
        ),
        qdrant_api_key: get_env_optional(&["QDRANT_API_KEY"]),
        chromadb_url: get_env_or(
            &["CHROMADB_URL"],
            defaults.chromadb_url,
        ),
        chromadb_collection: get_env_or(
            &["CHROMADB_COLLECTION"],
            defaults.chromadb_collection,
        ),

        // Caching
        cache_max_age_secs: get_env_parse_or(
            &["CACHE_MAX_AGE_SECS"],
            defaults.cache_max_age_secs,
        )?,
        cache_max_entries: get_env_parse_or(
            &["CACHE_MAX_ENTRIES"],
            defaults.cache_max_entries,
        )?,

        // Server
        server_host: get_env_or(&["SERVER_HOST"], defaults.server_host),
        server_port: get_env_parse_or(&["SERVER_PORT"], defaults.server_port)?,

        // OpenTelemetry
        enable_tracing: get_env_bool_or(&["ENABLE_TRACING"], defaults.enable_tracing)?,
        otel_endpoint: get_env_optional(&["OTEL_ENDPOINT"]),

        // Feature flags
        features: FeatureFlags {
            enable_pdf_ocr: get_env_bool_or(
                &["FEATURE_PDF_OCR"],
                defaults.features.enable_pdf_ocr,
            )?,
            enable_reranking: get_env_bool_or(
                &["FEATURE_RERANKING"],
                defaults.features.enable_reranking,
            )?,
            enable_section_chunking: get_env_bool_or(
                &["FEATURE_SECTION_CHUNKING"],
                defaults.features.enable_section_chunking,
            )?,
            enable_citation_formatting: get_env_bool_or(
                &["FEATURE_CITATION_FORMATTING"],
                defaults.features.enable_citation_formatting,
            )?,
            enable_context_compression: get_env_bool_or(
                &["FEATURE_CONTEXT_COMPRESSION"],
                defaults.features.enable_context_compression,
            )?,
            enable_extractive_summary: get_env_bool_or(
                &["FEATURE_EXTRACTIVE_SUMMARY"],
                defaults.features.enable_extractive_summary,
            )?,
        },
    };

    config.validate()?;
    Ok(config)
}

/// Get environment variable as string or use default
fn get_env_value(keys: &[&str]) -> Option<(String, String)> {
    keys.iter().find_map(|key| env::var(key).ok().map(|value| ((*key).to_string(), value)))
}

/// Get environment variable as string or use default
fn get_env_or(keys: &[&str], default: String) -> String {
    get_env_value(keys)
        .map(|(_, value)| value)
        .unwrap_or(default)
}

/// Get environment variable as optional string
fn get_env_optional(keys: &[&str]) -> Option<String> {
    get_env_value(keys).map(|(_, value)| value)
}

/// Get environment variable as PathBuf or use default
fn get_env_path_or(keys: &[&str], default: PathBuf) -> PathBuf {
    get_env_value(keys)
        .map(|(_, value)| PathBuf::from(value))
        .unwrap_or(default)
}

/// Parse environment variable or use default
fn get_env_parse_or<T>(keys: &[&str], default: T) -> Result<T>
where
    T: FromStr + Clone,
    T::Err: std::fmt::Display,
{
    match get_env_value(keys) {
        Some((key, val)) => val.parse::<T>().map_err(|e| {
            BbtError::Schema(format!("failed to parse env var {}: {}", key, e))
        }),
        None => Ok(default),
    }
}

/// Parse boolean environment variable or use default
fn get_env_bool_or(keys: &[&str], default: bool) -> Result<bool> {
    match get_env_value(keys) {
        Some((key, val)) => {
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
        None => Ok(default),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RetrievalMode;
    use std::sync::Mutex;

    // mutex to serialize tests that modify env vars
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn test_from_env_defaults() {
        let _guard = ENV_MUTEX.lock().unwrap();

        // Clear any existing env vars
        std::env::remove_var("CHUNK_SIZE");
        std::env::remove_var("LOG_LEVEL");

        let config = from_env().unwrap();
        assert_eq!(config.chunk_size, 512);
        assert_eq!(config.log_level, "info");
    }

    #[test]
    fn test_env_parsing() {
        let _guard = ENV_MUTEX.lock().unwrap();

        // save original values
        let orig_chunk = std::env::var("CHUNK_SIZE").ok();
        let orig_ocr = std::env::var("ENABLE_OCR").ok();
        let orig_mode = std::env::var("RETRIEVAL_MODE").ok();

        // set test values
        std::env::set_var("CHUNK_SIZE", "1024");
        std::env::set_var("ENABLE_OCR", "true");
        std::env::set_var("RETRIEVAL_MODE", "vector");

        let config = from_env().unwrap();
        assert_eq!(config.chunk_size, 1024);
        assert_eq!(config.enable_ocr, true);
        assert_eq!(config.retrieval_mode, RetrievalMode::Vector);

        // restore original values
        match orig_chunk {
            Some(v) => std::env::set_var("CHUNK_SIZE", v),
            None => std::env::remove_var("CHUNK_SIZE"),
        }
        match orig_ocr {
            Some(v) => std::env::set_var("ENABLE_OCR", v),
            None => std::env::remove_var("ENABLE_OCR"),
        }
        match orig_mode {
            Some(v) => std::env::set_var("RETRIEVAL_MODE", v),
            None => std::env::remove_var("RETRIEVAL_MODE"),
        }
    }
}
