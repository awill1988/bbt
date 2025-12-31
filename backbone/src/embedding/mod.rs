pub mod batch;
pub mod downloader;
pub mod models;
pub mod onnx;

pub use batch::{AsyncBatchEmbedder, BatchEmbedder};
pub use downloader::ensure_onnx_model;
pub use models::EmbeddingModelInfo;
pub use onnx::{ExecutionProvider, OnnxEmbedder};

use crate::error::Result;

/// Trait for embedding generators
pub trait Embedder {
    /// Generate embeddings for a batch of texts
    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// Get embedding dimensions
    fn dimensions(&self) -> usize;
}

/// Mock embedder for testing - generates deterministic fake embeddings
#[cfg(test)]
pub struct MockEmbedder {
    dimensions: usize,
}

#[cfg(test)]
impl MockEmbedder {
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }
}

#[cfg(test)]
impl Embedder for MockEmbedder {
    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // generate deterministic embeddings based on text hash
        Ok(texts
            .iter()
            .map(|text| {
                // simple hash-based deterministic embedding
                let hash = text.bytes().fold(0u64, |acc, b| acc.wrapping_add(b as u64));
                (0..self.dimensions)
                    .map(|i| ((hash.wrapping_mul(i as u64 + 1) % 1000) as f32) / 1000.0)
                    .collect()
            })
            .collect())
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}
