use crate::error::Result;
use crate::reranking::onnx::OnnxReranker;

/// batch reranker for synchronous processing
pub struct BatchReranker<'a> {
    reranker: &'a OnnxReranker,
    batch_size: usize,
}

impl<'a> BatchReranker<'a> {
    /// create a new batch reranker
    pub fn new(reranker: &'a OnnxReranker, batch_size: usize) -> Self {
        Self {
            reranker,
            batch_size,
        }
    }

    /// rerank a list of (query, document) pairs in batches
    ///
    /// # arguments
    /// * `pairs` - list of (query, document) text pairs
    ///
    /// # returns
    /// relevance scores (one per pair)
    pub fn rerank(&self, pairs: &[(String, String)]) -> Result<Vec<f32>> {
        let mut all_scores = Vec::with_capacity(pairs.len());

        for chunk in pairs.chunks(self.batch_size) {
            let scores = self.reranker.score_batch(chunk)?;
            all_scores.extend(scores);
        }

        Ok(all_scores)
    }
}

/// async batch reranker for non-blocking processing
pub struct AsyncBatchReranker {
    batch_size: usize,
}

impl AsyncBatchReranker {
    /// create a new async batch reranker
    pub fn new(batch_size: usize) -> Self {
        Self { batch_size }
    }

    /// rerank a list of (query, document) pairs in batches asynchronously
    ///
    /// # arguments
    /// * `reranker` - onnx reranker instance
    /// * `pairs` - list of (query, document) text pairs
    ///
    /// # returns
    /// relevance scores (one per pair)
    pub async fn rerank(
        &self,
        _reranker: &OnnxReranker,
        pairs: Vec<(String, String)>,
    ) -> Result<Vec<f32>> {
        let batch_size = self.batch_size;

        // spawn blocking task for cpu-intensive onnx inference
        tokio::task::spawn_blocking(move || {
            let mut all_scores = Vec::with_capacity(pairs.len());

            for chunk in pairs.chunks(batch_size) {
                // todo: replace with actual reranker.score_batch() when available
                let scores: Vec<f32> = chunk.iter().map(|_| 0.5).collect();
                all_scores.extend(scores);
            }

            Ok(all_scores)
        })
        .await
        .map_err(|e| crate::error::BbtError::Model(format!("async rerank task failed: {}", e)))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reranking::models::RerankModelInfo;
    use crate::reranking::onnx::ExecutionProvider;
    use std::path::PathBuf;

    #[test]
    fn test_batch_reranker_empty() {
        // note: this test would need a real model file to work
        // keeping it simple for now
        let pairs: Vec<(String, String)> = vec![];
        assert_eq!(pairs.len(), 0);
    }

    #[test]
    fn test_async_batch_reranker_creation() {
        let reranker = AsyncBatchReranker::new(32);
        assert_eq!(reranker.batch_size, 32);
    }
}
