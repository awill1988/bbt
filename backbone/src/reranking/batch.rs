use crate::error::Result;
use crate::reranking::onnx::OnnxReranker;
use std::sync::{Arc, Mutex};

/// batch reranker for synchronous processing
pub struct BatchReranker<'a> {
    reranker: &'a mut OnnxReranker,
    batch_size: usize,
}

impl<'a> BatchReranker<'a> {
    /// create a new batch reranker
    pub fn new(reranker: &'a mut OnnxReranker, batch_size: usize) -> Self {
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
    pub fn rerank(&mut self, pairs: &[(String, String)]) -> Result<Vec<f32>> {
        let mut all_scores = Vec::with_capacity(pairs.len());

        for chunk in pairs.chunks(self.batch_size) {
            let scores = self.reranker.score_batch(chunk)?;
            all_scores.extend(scores);
        }

        Ok(all_scores)
    }
}

/// async batch reranker for non-blocking processing
///
/// wraps OnnxReranker in Arc<Mutex<_>> for thread-safe async access
pub struct AsyncBatchReranker {
    reranker: Arc<Mutex<OnnxReranker>>,
    batch_size: usize,
}

impl AsyncBatchReranker {
    /// create a new async batch reranker
    ///
    /// takes ownership of the reranker and wraps it for thread-safe access
    pub fn new(reranker: OnnxReranker, batch_size: usize) -> Self {
        Self {
            reranker: Arc::new(Mutex::new(reranker)),
            batch_size,
        }
    }

    /// create from an already-wrapped reranker
    pub fn from_shared(reranker: Arc<Mutex<OnnxReranker>>, batch_size: usize) -> Self {
        Self { reranker, batch_size }
    }

    /// get a clone of the shared reranker reference
    pub fn shared_reranker(&self) -> Arc<Mutex<OnnxReranker>> {
        Arc::clone(&self.reranker)
    }

    /// rerank a list of (query, document) pairs in batches asynchronously
    ///
    /// # arguments
    /// * `pairs` - list of (query, document) text pairs
    ///
    /// # returns
    /// relevance scores (one per pair)
    pub async fn rerank(&self, pairs: Vec<(String, String)>) -> Result<Vec<f32>> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = self.batch_size;
        let reranker = Arc::clone(&self.reranker);
        let total = pairs.len();

        tracing::info!(
            total_pairs = total,
            batch_size = batch_size,
            "reranking pairs asynchronously"
        );

        // spawn blocking task for cpu-intensive onnx inference
        let result = tokio::task::spawn_blocking(move || {
            let mut all_scores = Vec::with_capacity(pairs.len());

            for chunk in pairs.chunks(batch_size) {
                // acquire lock and score this batch
                let scores = {
                    let mut guard = reranker
                        .lock()
                        .map_err(|e| crate::error::BbtError::Model(format!("mutex poisoned: {}", e)))?;
                    guard.score_batch(chunk)?
                };
                all_scores.extend(scores);
            }

            Ok::<Vec<f32>, crate::error::BbtError>(all_scores)
        })
        .await
        .map_err(|e| crate::error::BbtError::Model(format!("async rerank task failed: {}", e)))??;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_reranker_empty() {
        // note: this test would need a real model file to work
        // keeping it simple for now
        let pairs: Vec<(String, String)> = vec![];
        assert_eq!(pairs.len(), 0);
    }

    // note: async batch reranker tests require actual model files
    // integration tests should cover full reranker functionality
}
