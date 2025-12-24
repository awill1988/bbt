use crate::error::{BbtError, Result};
use crate::embedding::models::EmbeddingModelInfo;
use std::path::Path;

/// ONNX execution provider
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProvider {
    /// CPU provider (always available)
    Cpu,
    /// CUDA provider (NVIDIA GPUs)
    Cuda,
}

impl ExecutionProvider {
    /// Detect available execution providers
    /// Returns the best available provider
    pub fn detect() -> Self {
        // check for CUDA availability
        #[cfg(all(target_os = "linux", feature = "cuda"))]
        {
            if Self::is_cuda_available() {
                return Self::Cuda;
            }
        }

        // default to CPU
        Self::Cpu
    }

    /// Check if CUDA is available
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    fn is_cuda_available() -> bool {
        // check for nvidia-smi or CUDA libraries
        std::process::Command::new("nvidia-smi")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    /// Get provider name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        }
    }
}

/// ONNX-based embedding generator
///
/// This is a placeholder implementation that defines the interface.
/// Full implementation requires:
/// - ONNX model files
/// - ort crate integration
/// - Tokenizer integration
pub struct OnnxEmbedder {
    model_info: EmbeddingModelInfo,
    provider: ExecutionProvider,
    _model_path: std::path::PathBuf,
}

impl OnnxEmbedder {
    /// Create a new ONNX embedder
    ///
    /// # Arguments
    /// * `model_path` - Path to the ONNX model file
    /// * `model_info` - Model metadata
    /// * `provider` - Execution provider to use
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        model_info: EmbeddingModelInfo,
        provider: ExecutionProvider,
    ) -> Result<Self> {
        let model_path = model_path.as_ref().to_path_buf();

        if !model_path.exists() {
            return Err(BbtError::Model(format!(
                "model file not found: {}",
                model_path.display()
            )));
        }

        tracing::info!(
            model = %model_info.repo_id,
            provider = %provider.as_str(),
            dimensions = model_info.dimensions,
            "initialized onnx embedder"
        );

        Ok(Self {
            model_info,
            provider,
            _model_path: model_path,
        })
    }

    /// Generate embeddings for a batch of texts
    ///
    /// # Arguments
    /// * `texts` - Texts to embed
    ///
    /// # Returns
    /// Vector of embedding vectors (one per input text)
    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        tracing::debug!(
            count = texts.len(),
            provider = %self.provider.as_str(),
            "generating embeddings"
        );

        // todo: implement actual ONNX inference
        // this is a placeholder that returns zero vectors
        // real implementation would:
        // 1. tokenize texts
        // 2. run ONNX inference
        // 3. extract embeddings from output
        // 4. apply pooling (mean pooling typically)

        let embeddings = texts
            .iter()
            .map(|_| vec![0.0; self.model_info.dimensions])
            .collect();

        Ok(embeddings)
    }

    /// Get embedding dimensions
    pub fn dimensions(&self) -> usize {
        self.model_info.dimensions
    }

    /// Get model info
    pub fn model_info(&self) -> &EmbeddingModelInfo {
        &self.model_info
    }

    /// Get execution provider
    pub fn provider(&self) -> ExecutionProvider {
        self.provider
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_execution_provider_detect() {
        let provider = ExecutionProvider::detect();
        // should always have at least CPU
        assert!(provider == ExecutionProvider::Cpu || provider == ExecutionProvider::Cuda);
    }

    #[test]
    fn test_execution_provider_as_str() {
        assert_eq!(ExecutionProvider::Cpu.as_str(), "cpu");
        assert_eq!(ExecutionProvider::Cuda.as_str(), "cuda");
    }

    #[test]
    fn test_onnx_embedder_new() -> Result<()> {
        let mut temp_file = NamedTempFile::new()?;
        temp_file.write_all(b"dummy model")?;
        temp_file.flush()?;

        let model_info = EmbeddingModelInfo::default_model();
        let embedder = OnnxEmbedder::new(
            temp_file.path(),
            model_info,
            ExecutionProvider::Cpu,
        )?;

        assert_eq!(embedder.dimensions(), 384);
        assert_eq!(embedder.provider(), ExecutionProvider::Cpu);

        Ok(())
    }

    #[test]
    fn test_onnx_embedder_nonexistent_model() {
        let model_info = EmbeddingModelInfo::default_model();
        let result = OnnxEmbedder::new(
            "/nonexistent/model.onnx",
            model_info,
            ExecutionProvider::Cpu,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_onnx_embedder_embed_empty() -> Result<()> {
        let mut temp_file = NamedTempFile::new()?;
        temp_file.write_all(b"dummy model")?;
        temp_file.flush()?;

        let model_info = EmbeddingModelInfo::default_model();
        let embedder = OnnxEmbedder::new(
            temp_file.path(),
            model_info,
            ExecutionProvider::Cpu,
        )?;

        let embeddings = embedder.embed(&[])?;
        assert_eq!(embeddings.len(), 0);

        Ok(())
    }

    #[test]
    fn test_onnx_embedder_embed() -> Result<()> {
        let mut temp_file = NamedTempFile::new()?;
        temp_file.write_all(b"dummy model")?;
        temp_file.flush()?;

        let model_info = EmbeddingModelInfo::default_model();
        let embedder = OnnxEmbedder::new(
            temp_file.path(),
            model_info,
            ExecutionProvider::Cpu,
        )?;

        let texts = vec!["hello world".to_string(), "test".to_string()];
        let embeddings = embedder.embed(&texts)?;

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 384);
        assert_eq!(embeddings[1].len(), 384);

        Ok(())
    }
}
