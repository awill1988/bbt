use crate::error::{BbtError, Result};
use crate::embedding::models::EmbeddingModelInfo;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use tokenizers::Tokenizer;

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
        // check for CUDA availability on Linux
        #[cfg(target_os = "linux")]
        {
            if Self::is_cuda_available() {
                return Self::Cuda;
            }
        }

        // default to CPU
        Self::Cpu
    }

    /// Check if CUDA is available
    #[cfg(target_os = "linux")]
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
pub struct OnnxEmbedder {
    session: Session,
    tokenizer: Tokenizer,
    model_info: EmbeddingModelInfo,
    provider: ExecutionProvider,
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
        let model_path = model_path.as_ref();

        if !model_path.exists() {
            return Err(BbtError::Model(format!(
                "model file not found: {}",
                model_path.display()
            )));
        }

        // load onnx session with minimal settings to prevent deadlocks and hangs
        // disable all optimizations and force single-threaded execution
        let session = Session::builder()
            .map_err(|e| BbtError::Model(format!("failed to create session builder: {}", e)))?
            .with_optimization_level(GraphOptimizationLevel::Disable)
            .map_err(|e| BbtError::Model(format!("failed to set optimization level: {}", e)))?
            .with_intra_threads(1)
            .map_err(|e| BbtError::Model(format!("failed to set intra threads: {}", e)))?
            .with_inter_threads(1)
            .map_err(|e| BbtError::Model(format!("failed to set inter threads: {}", e)))?
            .commit_from_file(model_path)
            .map_err(|e| BbtError::Model(format!("failed to load model from {:?}: {}", model_path, e)))?;

        // load tokenizer (expects tokenizer.json in same directory as model)
        let tokenizer_path = model_path.parent()
            .ok_or_else(|| BbtError::Model("invalid model path".to_string()))?
            .join("tokenizer.json");

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| BbtError::Model(format!("failed to load tokenizer from {:?}: {}", tokenizer_path, e)))?;

        tracing::info!(
            model = %model_info.repo_id,
            provider = %provider.as_str(),
            dimensions = model_info.dimensions,
            "initialized onnx embedder"
        );

        Ok(Self {
            session,
            tokenizer,
            model_info,
            provider,
        })
    }

    /// Generate embeddings for a batch of texts
    ///
    /// # Arguments
    /// * `texts` - Texts to embed
    ///
    /// # Returns
    /// Vector of embedding vectors (one per input text)
    pub fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        tracing::debug!(
            count = texts.len(),
            provider = %self.provider.as_str(),
            "generating embeddings"
        );

        // tokenize all texts
        let encodings = self.tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| BbtError::Embedding(format!("tokenization failed: {}", e)))?;

        // prepare input tensors
        let batch_size = encodings.len();
        // cap max_len to model's maximum sequence length to avoid ONNX errors
        let model_max_len = self.model_info.max_seq_len;
        let max_len = encodings
            .iter()
            .map(|e| e.len())
            .max()
            .unwrap_or(0)
            .min(model_max_len);

        tracing::debug!(
            "batch_size={}, max_len={}, model_max_len={}",
            batch_size,
            max_len,
            model_max_len
        );

        // create input_ids tensor with padding
        let mut input_ids_vec = Vec::with_capacity(batch_size * max_len);
        let mut attention_mask_vec = Vec::with_capacity(batch_size * max_len);

        for encoding in &encodings {
            let ids = encoding.get_ids();
            let attention = encoding.get_attention_mask();
            let seq_len = ids.len();

            // truncate if exceeds max_len
            let truncated_len = seq_len.min(max_len);

            // add the actual tokens (truncated if necessary)
            input_ids_vec.extend_from_slice(&ids[..truncated_len]);
            attention_mask_vec.extend_from_slice(&attention[..truncated_len]);

            // pad to max_len with zeros (padding token id is typically 0)
            if truncated_len < max_len {
                input_ids_vec.resize(input_ids_vec.len() + (max_len - truncated_len), 0);
                attention_mask_vec.resize(attention_mask_vec.len() + (max_len - truncated_len), 0);
            }
        }

        // convert to i64 for onnx
        let input_ids: Vec<i64> = input_ids_vec.iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = attention_mask_vec.iter().map(|&x| x as i64).collect();

        // create onnx tensors
        let input_ids_tensor = Tensor::from_array((vec![batch_size, max_len], input_ids.into_boxed_slice()))
            .map_err(|e| BbtError::Embedding(format!("failed to create input_ids tensor: {}", e)))?;

        let attention_mask_tensor = Tensor::from_array((vec![batch_size, max_len], attention_mask.into_boxed_slice()))
            .map_err(|e| BbtError::Embedding(format!("failed to create attention_mask tensor: {}", e)))?;

        // check if model expects token_type_ids (bert-based models only)
        let requires_token_type_ids = self.session
            .inputs
            .iter()
            .any(|input| input.name == "token_type_ids");

        // run inference with appropriate inputs
        let outputs = if requires_token_type_ids {
            // bert-based models (bge-small, etc.)
            let token_type_ids: Vec<i64> = vec![0i64; batch_size * max_len];
            let token_type_ids_tensor = Tensor::from_array((vec![batch_size, max_len], token_type_ids.into_boxed_slice()))
                .map_err(|e| BbtError::Embedding(format!("failed to create token_type_ids tensor: {}", e)))?;

            self.session
                .run(ort::inputs![
                    "input_ids" => input_ids_tensor,
                    "attention_mask" => attention_mask_tensor,
                    "token_type_ids" => token_type_ids_tensor
                ])
                .map_err(|e| BbtError::Embedding(format!("onnx inference failed: {}", e)))?
        } else {
            // modern models (jina, etc.) - no token_type_ids
            self.session
                .run(ort::inputs![
                    "input_ids" => input_ids_tensor,
                    "attention_mask" => attention_mask_tensor
                ])
                .map_err(|e| BbtError::Embedding(format!("onnx inference failed: {}", e)))?
        };

        // extract last hidden state (output is typically "last_hidden_state")
        let output_tensor = outputs.get("last_hidden_state")
            .or_else(|| outputs.get("logits"))
            .ok_or_else(|| BbtError::Embedding("no output from model".to_string()))?;

        let (shape, output_data) = output_tensor.try_extract_tensor::<f32>()
            .map_err(|e| BbtError::Embedding(format!("failed to extract output tensor: {}", e)))?;

        if shape.len() != 3 {
            return Err(BbtError::Embedding(format!(
                "unexpected output shape: {:?}, expected [batch, seq_len, hidden_size]",
                shape
            )));
        }

        let hidden_size = shape[2] as usize;
        let seq_len = shape[1] as usize;

        // apply mean pooling with attention mask
        let mut embeddings = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let mut pooled = vec![0.0f32; hidden_size];
            let mut sum_mask = 0.0f32;

            // get attention mask for this batch item
            let mask_start = i * max_len;
            let mask_end = mask_start + seq_len;
            let mask_slice = &attention_mask_vec[mask_start..mask_end];

            for j in 0..seq_len {
                if mask_slice[j] == 1 {
                    // calculate flat index: [batch, seq, hidden] -> batch * (seq_len * hidden_size) + seq * hidden_size + hidden
                    let base_idx = (i * seq_len + j) * hidden_size;
                    for k in 0..hidden_size {
                        pooled[k] += output_data[base_idx + k];
                    }
                    sum_mask += 1.0;
                }
            }

            // average
            if sum_mask > 0.0 {
                for val in &mut pooled {
                    *val /= sum_mask;
                }
            }

            // normalize (l2 normalization for BGE models)
            let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in &mut pooled {
                    *val /= norm;
                }
            }

            embeddings.push(pooled);
        }

        tracing::debug!(
            "generated {} embeddings with dimension {}",
            embeddings.len(),
            self.model_info.dimensions
        );

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
