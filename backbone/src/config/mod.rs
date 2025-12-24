mod defaults;
mod env;
mod feature_flags;

pub use feature_flags::FeatureFlags;

use crate::error::Result;
use std::path::PathBuf;

/// Main configuration struct for bbt
#[derive(Debug, Clone)]
pub struct BbtConfig {
    // General
    pub log_level: String,

    // Paths
    pub data_dir: PathBuf,
    pub model_cache_dir: PathBuf,
    pub vector_store_path: PathBuf,
    pub bm25_index_path: PathBuf,
    pub state_store_path: PathBuf,

    // Document processing
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub enable_ocr: bool,
    pub enable_section_detection: bool,
    pub enable_header_footer_removal: bool,

    // Embedding
    pub embedding_model_repo: String,
    pub embedding_model_file: String,
    pub embedding_batch_size: usize,
    pub embedding_dims: usize,

    // Retrieval
    pub retrieval_mode: RetrievalMode,
    pub top_k: usize,
    pub min_score: f32,
    pub vector_weight: f32,
    pub bm25_weight: f32,
    pub enable_rerank: bool,
    pub rerank_model_repo: String,
    pub rerank_model_file: String,
    pub rerank_top_k: usize,

    // BM25
    pub bm25_k1: f32,
    pub bm25_b: f32,

    // Vector store
    pub vector_store_type: VectorStoreType,
    pub qdrant_url: String,
    pub qdrant_api_key: Option<String>,
    pub chromadb_url: String,
    pub chromadb_collection: String,

    // Caching
    pub cache_max_age_secs: u64,
    pub cache_max_entries: usize,

    // Server
    pub server_host: String,
    pub server_port: u16,

    // OpenTelemetry
    pub enable_tracing: bool,
    pub otel_endpoint: Option<String>,

    // Feature flags
    pub features: FeatureFlags,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalMode {
    Vector,
    Bm25,
    Hybrid,
}

impl std::str::FromStr for RetrievalMode {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "vector" => Ok(RetrievalMode::Vector),
            "bm25" => Ok(RetrievalMode::Bm25),
            "hybrid" => Ok(RetrievalMode::Hybrid),
            _ => Err(format!("invalid retrieval mode: {}", s)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorStoreType {
    Qdrant,
    ChromaDb,
}

impl std::str::FromStr for VectorStoreType {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "qdrant" => Ok(VectorStoreType::Qdrant),
            "chromadb" | "chroma" => Ok(VectorStoreType::ChromaDb),
            _ => Err(format!("invalid vector store type: {}", s)),
        }
    }
}

impl BbtConfig {
    /// Create configuration from environment variables
    pub fn from_env() -> Result<Self> {
        env::from_env()
    }

    /// Create default configuration
    pub fn default() -> Self {
        defaults::default_config()
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // chunk size validation
        if self.chunk_size == 0 {
            return Err(crate::error::BbtError::Schema(
                "chunk_size must be greater than 0".to_string(),
            ));
        }

        // chunk overlap validation
        if self.chunk_overlap >= self.chunk_size {
            return Err(crate::error::BbtError::Schema(
                "chunk_overlap must be less than chunk_size".to_string(),
            ));
        }

        // retrieval weights validation (for hybrid mode)
        if self.retrieval_mode == RetrievalMode::Hybrid {
            let weight_sum = self.vector_weight + self.bm25_weight;
            if (weight_sum - 1.0).abs() > 0.001 {
                return Err(crate::error::BbtError::Schema(format!(
                    "vector_weight + bm25_weight must equal 1.0, got {}",
                    weight_sum
                )));
            }
        }

        // BM25 parameters validation
        if self.bm25_k1 <= 0.0 {
            return Err(crate::error::BbtError::Schema(
                "bm25_k1 must be greater than 0".to_string(),
            ));
        }
        if self.bm25_b < 0.0 || self.bm25_b > 1.0 {
            return Err(crate::error::BbtError::Schema(
                "bm25_b must be between 0 and 1".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_validation() {
        let config = BbtConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_retrieval_mode_parsing() {
        assert_eq!("vector".parse::<RetrievalMode>().unwrap(), RetrievalMode::Vector);
        assert_eq!("bm25".parse::<RetrievalMode>().unwrap(), RetrievalMode::Bm25);
        assert_eq!("hybrid".parse::<RetrievalMode>().unwrap(), RetrievalMode::Hybrid);
        assert!("invalid".parse::<RetrievalMode>().is_err());
    }

    #[test]
    fn test_vector_store_type_parsing() {
        assert_eq!("qdrant".parse::<VectorStoreType>().unwrap(), VectorStoreType::Qdrant);
        assert_eq!("chromadb".parse::<VectorStoreType>().unwrap(), VectorStoreType::ChromaDb);
        assert_eq!("chroma".parse::<VectorStoreType>().unwrap(), VectorStoreType::ChromaDb);
        assert!("invalid".parse::<VectorStoreType>().is_err());
    }
}
