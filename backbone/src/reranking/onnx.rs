use crate::error::{BbtError, Result};
use crate::reranking::models::RerankModelInfo;
use ort::execution_providers::{
    CoreMLExecutionProvider, CPUExecutionProvider, CUDAExecutionProvider,
    ExecutionProvider as OrtExecutionProvider,
};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use std::env;
use std::path::Path;

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

fn is_force_cpu() -> bool {
    env::var("BBT_FORCE_CPU")
        .map(|v| {
            let v = v.to_lowercase();
            v == "1" || v == "true" || v == "yes"
        })
        .unwrap_or(false)
}

/// onnx-based cross-encoder reranker
pub struct OnnxReranker {
    #[allow(dead_code)]
    session: Session,
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
        let session_builder = Session::builder()
            .map_err(|e| BbtError::Model(format!("failed to create session builder: {}", e)))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| BbtError::Model(format!("failed to set optimization level: {}", e)))?;

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
    pub fn score_batch(&self, pairs: &[(String, String)]) -> Result<Vec<f32>> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }

        // todo: implement actual onnx inference
        // for now, return placeholder scores based on text similarity
        tracing::warn!(
            "onnx reranker inference not yet implemented, returning placeholder scores"
        );

        let scores: Vec<f32> = pairs
            .iter()
            .map(|(query, doc)| {
                // simple heuristic: score based on common words
                let query_lower = query.to_lowercase();
                let doc_lower = doc.to_lowercase();
                let query_words: std::collections::HashSet<_> =
                    query_lower.split_whitespace().collect();
                let doc_words: std::collections::HashSet<_> =
                    doc_lower.split_whitespace().collect();

                let common = query_words.intersection(&doc_words).count() as f32;
                let total = query_words.len().max(doc_words.len()) as f32;

                if total > 0.0 {
                    common / total
                } else {
                    0.0
                }
            })
            .collect();

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
