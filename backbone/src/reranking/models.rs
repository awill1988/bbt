/// model metadata for cross-encoder reranking models
#[derive(Debug, Clone)]
pub struct RerankModelInfo {
    /// huggingface repository id
    pub repo_id: String,

    /// onnx model filename within the repo
    pub model_file: String,

    /// maximum sequence length (tokens)
    pub max_seq_len: usize,

    /// whether to normalize scores via sigmoid
    pub normalize_scores: bool,

    /// model description
    pub description: String,
}

impl RerankModelInfo {
    /// default rerank model: cross-encoder/ms-marco-minilm-l-6-v2
    pub fn default_model() -> Self {
        Self {
            repo_id: "cross-encoder/ms-marco-MiniLM-L-6-v2".to_string(),
            model_file: "onnx/model.onnx".to_string(),
            max_seq_len: 512,
            normalize_scores: true,
            description: "cross-encoder minilm l-6 v2 - fast and accurate reranking".to_string(),
        }
    }

    /// nvidia nemotron reranker (requires gpu)
    pub fn nvidia_rerankqa_mistral_4b() -> Self {
        Self {
            repo_id: "nvidia/nv-rerankqa-mistral-4b-v3".to_string(),
            model_file: "onnx/model.onnx".to_string(),
            max_seq_len: 2048,
            normalize_scores: true,
            description: "nvidia rerankqa mistral 4b v3 - state-of-the-art reranking".to_string(),
        }
    }

    /// create custom rerank model info
    pub fn custom(
        repo_id: String,
        model_file: String,
        max_seq_len: usize,
        normalize_scores: bool,
    ) -> Self {
        Self {
            repo_id: repo_id.clone(),
            model_file,
            max_seq_len,
            normalize_scores,
            description: format!("custom rerank model: {}", repo_id),
        }
    }
}

impl Default for RerankModelInfo {
    fn default() -> Self {
        Self::default_model()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_model() {
        let model = RerankModelInfo::default_model();
        assert_eq!(model.repo_id, "cross-encoder/ms-marco-MiniLM-L-6-v2");
        assert_eq!(model.max_seq_len, 512);
        assert!(model.normalize_scores);
    }

    #[test]
    fn test_nvidia_model() {
        let model = RerankModelInfo::nvidia_rerankqa_mistral_4b();
        assert!(model.repo_id.contains("nvidia"));
        assert_eq!(model.max_seq_len, 2048);
    }

    #[test]
    fn test_custom_model() {
        let model = RerankModelInfo::custom(
            "test/reranker".to_string(),
            "model.onnx".to_string(),
            1024,
            false,
        );
        assert_eq!(model.repo_id, "test/reranker");
        assert_eq!(model.max_seq_len, 1024);
        assert!(!model.normalize_scores);
    }
}
