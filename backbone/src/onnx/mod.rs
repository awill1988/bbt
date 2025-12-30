pub mod gpu_capabilities;

use ort::execution_providers::{CUDAExecutionProvider, ExecutionProvider as ExecutionProviderTrait};

#[cfg(target_vendor = "apple")]
use ort::execution_providers::CoreMLExecutionProvider;
use std::env;

pub use gpu_capabilities::{calculate_worker_config, GpuCapabilities, WorkerConfig};

/// ONNX execution provider
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProvider {
    /// CPU provider (always available)
    Cpu,
    /// CoreML provider (macOS)
    Coreml,
    /// CUDA provider (NVIDIA GPUs)
    Cuda,
}

impl ExecutionProvider {
    /// Detect available execution providers
    /// Returns the best available provider
    pub fn detect() -> Self {
        if is_force_cpu() {
            tracing::info!("gpu acceleration disabled (cpu mode forced)");
            return Self::Cpu;
        }

        #[cfg(target_vendor = "apple")]
        {
            if cfg!(feature = "coreml")
                && CoreMLExecutionProvider::default()
                    .is_available()
                    .unwrap_or(false)
            {
                return Self::Coreml;
            }
        }

        if cfg!(feature = "cuda")
            && CUDAExecutionProvider::default()
                .is_available()
                .unwrap_or(false)
        {
            return Self::Cuda;
        }

        Self::Cpu
    }

    /// Get provider name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Coreml => "coreml",
            Self::Cuda => "cuda",
        }
    }
}

/// Check if CPU mode is forced via environment variables
/// Checks multiple env vars for backward compatibility
fn is_force_cpu() -> bool {
    // check embedding-specific var first
    if env::var("EMBEDDING_FORCE_CPU")
        .map(|v| {
            let v = v.to_lowercase();
            v == "1" || v == "true" || v == "yes"
        })
        .unwrap_or(false)
    {
        return true;
    }

    // check reranking-specific var
    if env::var("RERANK_FORCE_CPU")
        .map(|v| {
            let v = v.to_lowercase();
            v == "1" || v == "true" || v == "yes"
        })
        .unwrap_or(false)
    {
        return true;
    }

    // check global var
    env::var("LLM_FORCE_CPU")
        .map(|v| {
            let v = v.to_lowercase();
            v == "1" || v == "true" || v == "yes"
        })
        .unwrap_or(false)
}
