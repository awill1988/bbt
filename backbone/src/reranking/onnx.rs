use crate::error::{BbtError, Result};
use crate::reranking::models::RerankModelInfo;
use ort::execution_providers::{
    CoreMLExecutionProvider, CPUExecutionProvider, CUDAExecutionProvider,
    ExecutionProvider as OrtExecutionProvider,
};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::env;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;

/// execution provider for onnx runtime
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProvider {
    Cpu,
    Coreml,
    Cuda,
}

impl ExecutionProvider {
    /// detect best available execution provider
    pub fn detect() -> Self {
        if is_force_cpu() {
            tracing::info!("gpu acceleration disabled (cpu mode forced)");
            return ExecutionProvider::Cpu;
        }

        #[cfg(target_vendor = "apple")]
        {
            if cfg!(feature = "coreml")
                && CoreMLExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false)
            {
                return ExecutionProvider::Coreml;
            }
        }

        if cfg!(feature = "cuda")
            && CUDAExecutionProvider::default()
                .is_available()
                .unwrap_or(false)
        {
            ExecutionProvider::Cuda
        } else {
            ExecutionProvider::Cpu
        }
    }
}

fn read_bool_env(keys: &[&str]) -> bool {
    for key in keys {
        if let Ok(value) = env::var(key) {
            let value = value.to_lowercase();
            return value == "1" || value == "true" || value == "yes";
        }
    }
    false
}

fn is_force_cpu() -> bool {
    read_bool_env(&["FORCE_CPU"])
}

/// onnx-based cross-encoder reranker
pub struct OnnxReranker {
    session: Session,
    tokenizer: Arc<Tokenizer>,
    model_info: RerankModelInfo,
    provider: ExecutionProvider,
}

impl OnnxReranker {
    /// create a new onnx reranker
    ///
    /// # arguments
    /// * `model_path` - path to onnx model file
    /// * `model_info` - model metadata
    /// * `provider` - execution provider (cpu/coreml/cuda)
    pub fn new(
        model_path: &Path,
        model_info: RerankModelInfo,
        provider: ExecutionProvider,
    ) -> Result<Self> {
        // load tokenizer (expects tokenizer.json in same directory as model)
        let tokenizer_path = model_path.parent()
            .ok_or_else(|| BbtError::Model("invalid model path".to_string()))?
            .join("tokenizer.json");

        let tokenizer = Arc::new(
            Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| BbtError::Model(format!("failed to load tokenizer from {:?}: {}", tokenizer_path, e)))?
        );

        Self::new_with_tokenizer(model_path, model_info, provider, tokenizer)
    }

    /// create a new onnx reranker with a shared tokenizer
    ///
    /// # arguments
    /// * `model_path` - path to onnx model file
    /// * `model_info` - model metadata
    /// * `provider` - execution provider (cpu/coreml/cuda)
    /// * `tokenizer` - shared tokenizer instance
    pub fn new_with_tokenizer(
        model_path: &Path,
        model_info: RerankModelInfo,
        provider: ExecutionProvider,
        tokenizer: Arc<Tokenizer>,
    ) -> Result<Self> {
        let session_builder = Session::builder()
            .map_err(|e| BbtError::Model(format!("failed to create session builder: {}", e)))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| BbtError::Model(format!("failed to set optimization level: {}", e)))?
            .with_intra_threads(1)
            .map_err(|e| BbtError::Model(format!("failed to set intra threads: {}", e)))?
            .with_inter_threads(1)
            .map_err(|e| BbtError::Model(format!("failed to set inter threads: {}", e)))?;

        let session_builder = match provider {
            ExecutionProvider::Cuda => session_builder
                .with_execution_providers([
                    CUDAExecutionProvider::default().build(),
                    CPUExecutionProvider::default().build(),
                ])
                .map_err(|e| {
                    BbtError::Model(format!("failed to set execution providers: {}", e))
                })?,
            ExecutionProvider::Coreml => session_builder
                .with_execution_providers([
                    CoreMLExecutionProvider::default().build(),
                    CPUExecutionProvider::default().build(),
                ])
                .map_err(|e| {
                    BbtError::Model(format!("failed to set execution providers: {}", e))
                })?,
            ExecutionProvider::Cpu => session_builder
                .with_execution_providers([CPUExecutionProvider::default().build()])
                .map_err(|e| {
                    BbtError::Model(format!("failed to set execution providers: {}", e))
                })?,
        };

        let session = session_builder
            .commit_from_file(model_path)
            .map_err(|e| BbtError::Model(format!("failed to load model from {:?}: {}", model_path, e)))?;

        tracing::info!(
            "initialized onnx reranker model={} provider={:?} max_len={}",
            model_info.repo_id,
            provider,
            model_info.max_seq_len
        );

        Ok(Self {
            session,
            tokenizer,
            model_info,
            provider,
        })
    }

    /// score a batch of (query, document) pairs
    ///
    /// # arguments
    /// * `pairs` - list of (query, document) text pairs
    ///
    /// # returns
    /// relevance scores (one per pair)
    pub fn score_batch(&mut self, pairs: &[(String, String)]) -> Result<Vec<f32>> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }

        tracing::debug!(
            count = pairs.len(),
            provider = ?self.provider,
            "scoring query-document pairs"
        );

        // for cross-encoders, we encode each (query, doc) pair together
        // the tokenizer handles the [CLS] query [SEP] doc [SEP] format
        let texts: Vec<(&str, &str)> = pairs
            .iter()
            .map(|(q, d)| (q.as_str(), d.as_str()))
            .collect();

        let encodings = self.tokenizer
            .encode_batch(texts, true)
            .map_err(|e| BbtError::Model(format!("tokenization failed: {}", e)))?;

        // prepare input tensors
        let batch_size = encodings.len();
        let model_max_len = self.model_info.max_seq_len;
        let max_len = encodings
            .iter()
            .map(|e| e.len())
            .max()
            .unwrap_or(0)
            .min(model_max_len);

        // create input tensors with padding
        let mut input_ids_vec = Vec::with_capacity(batch_size * max_len);
        let mut attention_mask_vec = Vec::with_capacity(batch_size * max_len);

        for encoding in &encodings {
            let ids = encoding.get_ids();
            let attention = encoding.get_attention_mask();
            let seq_len = ids.len();
            let truncated_len = seq_len.min(max_len);

            input_ids_vec.extend_from_slice(&ids[..truncated_len]);
            attention_mask_vec.extend_from_slice(&attention[..truncated_len]);

            // pad to max_len
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
            .map_err(|e| BbtError::Model(format!("failed to create input_ids tensor: {}", e)))?;

        let attention_mask_tensor = Tensor::from_array((vec![batch_size, max_len], attention_mask.into_boxed_slice()))
            .map_err(|e| BbtError::Model(format!("failed to create attention_mask tensor: {}", e)))?;

        // check if model expects token_type_ids
        let requires_token_type_ids = self.session
            .inputs
            .iter()
            .any(|input| input.name == "token_type_ids");

        // capture output names before mutable borrow for error messages
        let output_names: Vec<_> = self.session.outputs.iter().map(|o| o.name.clone()).collect();

        // run inference
        let outputs = if requires_token_type_ids {
            let token_type_ids: Vec<i64> = vec![0i64; batch_size * max_len];
            let token_type_ids_tensor = Tensor::from_array((vec![batch_size, max_len], token_type_ids.into_boxed_slice()))
                .map_err(|e| BbtError::Model(format!("failed to create token_type_ids tensor: {}", e)))?;

            self.session
                .run(ort::inputs![
                    "input_ids" => input_ids_tensor,
                    "attention_mask" => attention_mask_tensor,
                    "token_type_ids" => token_type_ids_tensor
                ])
                .map_err(|e| BbtError::Model(format!("onnx inference failed: {}", e)))?
        } else {
            self.session
                .run(ort::inputs![
                    "input_ids" => input_ids_tensor,
                    "attention_mask" => attention_mask_tensor
                ])
                .map_err(|e| BbtError::Model(format!("onnx inference failed: {}", e)))?
        };

        // extract scores from output
        // cross-encoder models typically output logits with shape [batch_size, num_labels]
        // for binary relevance, num_labels=1 or num_labels=2
        // try common output names first, then fall back to index-based access
        let output_tensor = outputs.get("logits")
            .or_else(|| outputs.get("output"))
            .or_else(|| outputs.get("scores"))
            .or_else(|| outputs.get("output_0"))
            .ok_or_else(|| {
                BbtError::Model(format!("no output from reranker model, available: {:?}", output_names))
            })?;

        let (shape, output_data) = output_tensor.try_extract_tensor::<f32>()
            .map_err(|e| BbtError::Model(format!("failed to extract output tensor: {}", e)))?;

        // extract scores based on output shape
        let scores: Vec<f32> = if shape.len() == 2 && shape[1] == 1 {
            // [batch_size, 1] - single logit per pair
            output_data.iter().copied().collect()
        } else if shape.len() == 2 && shape[1] == 2 {
            // [batch_size, 2] - binary classification, use logit for positive class
            output_data.chunks(2).map(|chunk| chunk[1]).collect()
        } else if shape.len() == 1 {
            // [batch_size] - already flattened scores
            output_data.iter().copied().collect()
        } else {
            return Err(BbtError::Model(format!(
                "unexpected output shape: {:?}, expected [batch_size, 1] or [batch_size, 2]",
                shape
            )));
        };

        tracing::debug!(
            "generated {} reranking scores",
            scores.len()
        );

        // apply sigmoid normalization if configured
        if self.model_info.normalize_scores {
            Ok(scores.iter().map(|&s| sigmoid(s)).collect())
        } else {
            Ok(scores)
        }
    }

    /// get model information
    pub fn model_info(&self) -> &RerankModelInfo {
        &self.model_info
    }

    /// get execution provider
    pub fn provider(&self) -> ExecutionProvider {
        self.provider
    }
}

/// sigmoid normalization
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 0.001);
        assert!(sigmoid(10.0) > 0.9);
        assert!(sigmoid(-10.0) < 0.1);
    }

    #[test]
    fn test_execution_provider_detection() {
        let provider = ExecutionProvider::detect();
        // should return a supported provider
        assert!(matches!(
            provider,
            ExecutionProvider::Cpu | ExecutionProvider::Coreml | ExecutionProvider::Cuda
        ));
    }
}
