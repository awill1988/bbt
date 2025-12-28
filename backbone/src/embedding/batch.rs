use crate::error::Result;
use crate::embedding::onnx::OnnxEmbedder;

/// Batch embedding processor
pub struct BatchEmbedder<'a> {
    embedder: &'a mut OnnxEmbedder,
    batch_size: usize,
}

impl<'a> BatchEmbedder<'a> {
    /// Create a new batch embedder
    ///
    /// # Arguments
    /// * `embedder` - The ONNX embedder to use
    /// * `batch_size` - Number of texts to process per batch
    pub fn new(embedder: &'a mut OnnxEmbedder, batch_size: usize) -> Self {
        Self {
            embedder,
            batch_size,
        }
    }

    /// Process texts in batches and generate embeddings
    ///
    /// # Arguments
    /// * `texts` - All texts to embed
    ///
    /// # Returns
    /// Vector of embeddings (one per input text)
    pub fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let total = texts.len();
        let mut all_embeddings = Vec::with_capacity(total);

        tracing::info!(
            total_texts = total,
            batch_size = self.batch_size,
            batches = (total + self.batch_size - 1) / self.batch_size,
            "processing embeddings in batches"
        );

        // process in batches
        for (batch_idx, chunk) in texts.chunks(self.batch_size).enumerate() {
            tracing::debug!(
                batch = batch_idx,
                batch_size = chunk.len(),
                "processing batch"
            );

            let embeddings = self.embedder.embed(chunk)?;
            all_embeddings.extend(embeddings);
        }

        assert_eq!(
            all_embeddings.len(),
            total,
            "mismatch in embedding count"
        );

        Ok(all_embeddings)
    }

    /// Get batch size
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

/// Async batch embedding processor
pub struct AsyncBatchEmbedder<'a> {
    embedder: &'a OnnxEmbedder,
    batch_size: usize,
}

impl<'a> AsyncBatchEmbedder<'a> {
    /// Create a new async batch embedder
    pub fn new(embedder: &'a OnnxEmbedder, batch_size: usize) -> Self {
        Self {
            embedder,
            batch_size,
        }
    }

    /// Process texts in batches asynchronously
    ///
    /// Runs embedding generation in a blocking task to avoid blocking async runtime
    pub async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let batch_size = self.batch_size;
        let dimensions = self.embedder.dimensions();
        let total = texts.len();

        tracing::info!(
            total_texts = total,
            batch_size = batch_size,
            dimensions = dimensions,
            "processing embeddings asynchronously"
        );

        // run embedding in blocking task
        let result = tokio::task::spawn_blocking(move || {
            let mut all_embeddings = Vec::with_capacity(total);

            for chunk in texts.chunks(batch_size) {
                // simulate embedding generation
                // todo: replace with actual embedder.embed() when available
                let embeddings: Vec<Vec<f32>> = chunk
                    .iter()
                    .map(|_| vec![0.0; dimensions])
                    .collect();

                all_embeddings.extend(embeddings);
            }

            all_embeddings
        })
        .await
        .map_err(|e| crate::error::BbtError::Model(format!("task join error: {}", e)))?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::models::EmbeddingModelInfo;
    use crate::embedding::onnx::{ExecutionProvider, OnnxEmbedder};
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_embedder() -> Result<OnnxEmbedder> {
        let mut temp_file = NamedTempFile::new()?;
        temp_file.write_all(b"dummy model")?;
        temp_file.flush()?;

        let model_info = EmbeddingModelInfo::default_model();
        OnnxEmbedder::new(temp_file.path(), model_info, ExecutionProvider::Cpu)
    }

    #[test]
    fn test_batch_embedder_empty() -> Result<()> {
        let embedder = create_test_embedder()?;
        let batch_embedder = BatchEmbedder::new(&embedder, 32);

        let embeddings = batch_embedder.embed_batch(&[])?;
        assert_eq!(embeddings.len(), 0);

        Ok(())
    }

    #[test]
    fn test_batch_embedder_single_batch() -> Result<()> {
        let embedder = create_test_embedder()?;
        let batch_embedder = BatchEmbedder::new(&embedder, 32);

        let texts: Vec<String> = (0..10).map(|i| format!("text {}", i)).collect();
        let embeddings = batch_embedder.embed_batch(&texts)?;

        assert_eq!(embeddings.len(), 10);
        assert_eq!(embeddings[0].len(), 384);

        Ok(())
    }

    #[test]
    fn test_batch_embedder_multiple_batches() -> Result<()> {
        let embedder = create_test_embedder()?;
        let batch_embedder = BatchEmbedder::new(&embedder, 10);

        let texts: Vec<String> = (0..35).map(|i| format!("text {}", i)).collect();
        let embeddings = batch_embedder.embed_batch(&texts)?;

        // should process in 4 batches: 10, 10, 10, 5
        assert_eq!(embeddings.len(), 35);

        Ok(())
    }

    #[test]
    fn test_batch_embedder_batch_size() -> Result<()> {
        let embedder = create_test_embedder()?;
        let batch_embedder = BatchEmbedder::new(&embedder, 16);

        assert_eq!(batch_embedder.batch_size(), 16);

        Ok(())
    }

    #[tokio::test]
    async fn test_async_batch_embedder() -> Result<()> {
        let embedder = create_test_embedder()?;
        let async_embedder = AsyncBatchEmbedder::new(&embedder, 32);

        let texts: Vec<String> = (0..10).map(|i| format!("text {}", i)).collect();
        let embeddings = async_embedder.embed_batch(texts).await?;

        assert_eq!(embeddings.len(), 10);
        assert_eq!(embeddings[0].len(), 384);

        Ok(())
    }

    #[tokio::test]
    async fn test_async_batch_embedder_empty() -> Result<()> {
        let embedder = create_test_embedder()?;
        let async_embedder = AsyncBatchEmbedder::new(&embedder, 32);

        let embeddings = async_embedder.embed_batch(vec![]).await?;
        assert_eq!(embeddings.len(), 0);

        Ok(())
    }
}
