/// Model metadata for embedding models
#[derive(Debug, Clone)]
pub struct EmbeddingModelInfo {
    /// HuggingFace repository ID
    pub repo_id: String,

    /// ONNX model filename within the repo
    pub model_file: String,

    /// Embedding dimensions
    pub dimensions: usize,

    /// Maximum sequence length (tokens)
    pub max_seq_len: usize,

    /// Model description
    pub description: String,
}

impl EmbeddingModelInfo {
    /// Default embedding model: jina-embeddings-v2-base-code (optimized for code)
    pub fn default_model() -> Self {
        Self {
            repo_id: "jinaai/jina-embeddings-v2-base-code".to_string(),
            model_file: "onnx/model.onnx".to_string(),
            dimensions: 768,
            max_seq_len: 8192,
            description: "Jina Embeddings v2 Base Code - long-context code-optimized embeddings".to_string(),
        }
    }

    /// Legacy model: BAAI/bge-small-en-v1.5
    pub fn bge_small() -> Self {
        Self {
            repo_id: "BAAI/bge-small-en-v1.5".to_string(),
            model_file: "onnx/model.onnx".to_string(),
            dimensions: 384,
            max_seq_len: 512,
            description: "BAAI BGE Small English v1.5 - efficient general-purpose embeddings".to_string(),
        }
    }

    /// Alternative model: all-MiniLM-L6-v2
    pub fn minilm_l6() -> Self {
        Self {
            repo_id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            model_file: "onnx/model.onnx".to_string(),
            dimensions: 384,
            max_seq_len: 256,
            description: "Sentence Transformers MiniLM L6 v2 - fast and lightweight".to_string(),
        }
    }

    /// nvidia nemotron embedding model (requires onnx export)
    pub fn nemotron_8b() -> Self {
        Self {
            repo_id: "nvidia/llama-embed-nemotron-8b".to_string(),
            model_file: "onnx/model.onnx".to_string(),
            dimensions: 4096,
            max_seq_len: 4096,
            description: "nvidia nemotron 8b - state-of-the-art multilingual embeddings".to_string(),
        }
    }

    /// Create custom model info
    pub fn custom(repo_id: String, model_file: String, dimensions: usize, max_seq_len: usize) -> Self {
        Self {
            repo_id: repo_id.clone(),
            model_file,
            dimensions,
            max_seq_len,
            description: format!("Custom model: {}", repo_id),
        }
    }

    /// Apply a cap to max_seq_len for VRAM optimization
    /// Returns a new model info with max_seq_len capped to the specified value
    pub fn with_max_seq_len_cap(mut self, cap: usize) -> Self {
        if cap > 0 && cap < self.max_seq_len {
            tracing::info!(
                "capping max_seq_len from {} to {} for VRAM optimization",
                self.max_seq_len,
                cap
            );
            self.max_seq_len = cap;
        }
        self
    }
}

impl Default for EmbeddingModelInfo {
    fn default() -> Self {
        Self::default_model()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_model() {
        let model = EmbeddingModelInfo::default_model();
        assert_eq!(model.repo_id, "jinaai/jina-embeddings-v2-base-code");
        assert_eq!(model.dimensions, 768);
        assert_eq!(model.max_seq_len, 8192);
    }

    #[test]
    fn test_minilm_model() {
        let model = EmbeddingModelInfo::minilm_l6();
        assert_eq!(model.dimensions, 384);
        assert!(model.repo_id.contains("MiniLM"));
    }

    #[test]
    fn test_custom_model() {
        let model = EmbeddingModelInfo::custom(
            "test/model".to_string(),
            "model.onnx".to_string(),
            768,
            1024,
        );
        assert_eq!(model.repo_id, "test/model");
        assert_eq!(model.dimensions, 768);
        assert_eq!(model.max_seq_len, 1024);
    }

    #[test]
    fn test_max_seq_len_cap() {
        // cap should reduce max_seq_len when cap is smaller
        let model = EmbeddingModelInfo::default_model();
        assert_eq!(model.max_seq_len, 8192);

        let capped = model.with_max_seq_len_cap(2048);
        assert_eq!(capped.max_seq_len, 2048);
    }

    #[test]
    fn test_max_seq_len_cap_larger_than_model() {
        // cap larger than model max_seq_len should not increase it
        let model = EmbeddingModelInfo::minilm_l6();
        assert_eq!(model.max_seq_len, 256);

        let capped = model.with_max_seq_len_cap(4096);
        assert_eq!(capped.max_seq_len, 256); // unchanged
    }

    #[test]
    fn test_max_seq_len_cap_zero() {
        // cap of 0 should not change anything
        let model = EmbeddingModelInfo::default_model();
        let capped = model.with_max_seq_len_cap(0);
        assert_eq!(capped.max_seq_len, 8192); // unchanged
    }
}
