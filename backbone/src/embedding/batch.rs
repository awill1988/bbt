use crate::error::Result;
use crate::embedding::Embedder;

/// Batch embedding processor - generic over any Embedder implementation
pub struct BatchEmbedder<'a, E: Embedder> {
    embedder: &'a mut E,
    batch_size: usize,
}

impl<'a, E: Embedder> BatchEmbedder<'a, E> {
    /// Create a new batch embedder
    ///
    /// # Arguments
    /// * `embedder` - The embedder to use (implements Embedder trait)
    /// * `batch_size` - Number of texts to process per batch
    pub fn new(embedder: &'a mut E, batch_size: usize) -> Self {
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

    /// Get embedding dimensions from the underlying embedder
    pub fn dimensions(&self) -> usize {
        self.embedder.dimensions()
    }
}

/// Async batch embedding processor - generic over any Embedder implementation
///
/// Note: Current implementation uses spawn_blocking with placeholder embeddings.
/// For real async embedding, would need Arc<Mutex<E>> or Send + Sync embedder.
pub struct AsyncBatchEmbedder<'a, E: Embedder> {
    embedder: &'a E,
    batch_size: usize,
}

impl<'a, E: Embedder> AsyncBatchEmbedder<'a, E> {
    /// Create a new async batch embedder
    pub fn new(embedder: &'a E, batch_size: usize) -> Self {
        Self {
            embedder,
            batch_size,
        }
    }

    /// Get embedding dimensions from the underlying embedder
    pub fn dimensions(&self) -> usize {
        self.embedder.dimensions()
    }

    /// Process texts in batches asynchronously
    ///
    /// Runs embedding generation in a blocking task to avoid blocking async runtime.
    /// Note: Currently generates placeholder embeddings - real implementation would
    /// need Arc<Mutex<E>> to share embedder across thread boundary.
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
        // todo: use Arc<Mutex<E>> to actually call embedder.embed() in blocking task
        let result = tokio::task::spawn_blocking(move || {
            let mut all_embeddings = Vec::with_capacity(total);

            for chunk in texts.chunks(batch_size) {
                // placeholder: generate deterministic embeddings based on text
                let embeddings: Vec<Vec<f32>> = chunk
                    .iter()
                    .map(|text| {
                        let hash = text.bytes().fold(0u64, |acc, b| acc.wrapping_add(b as u64));
                        (0..dimensions)
                            .map(|i| ((hash.wrapping_mul(i as u64 + 1) % 1000) as f32) / 1000.0)
                            .collect()
                    })
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
    use crate::embedding::MockEmbedder;

    const TEST_DIMENSIONS: usize = 768;

    fn create_mock_embedder() -> MockEmbedder {
        MockEmbedder::new(TEST_DIMENSIONS)
    }

    #[test]
    fn test_batch_embedder_empty() -> Result<()> {
        let mut embedder = create_mock_embedder();
        let mut batch_embedder = BatchEmbedder::new(&mut embedder, 32);

        let embeddings = batch_embedder.embed_batch(&[])?;
        assert_eq!(embeddings.len(), 0);

        Ok(())
    }

    #[test]
    fn test_batch_embedder_single_batch() -> Result<()> {
        let mut embedder = create_mock_embedder();
        let mut batch_embedder = BatchEmbedder::new(&mut embedder, 32);

        let texts: Vec<String> = (0..10).map(|i| format!("text {}", i)).collect();
        let embeddings = batch_embedder.embed_batch(&texts)?;

        assert_eq!(embeddings.len(), 10);
        assert_eq!(embeddings[0].len(), TEST_DIMENSIONS);

        Ok(())
    }

    #[test]
    fn test_batch_embedder_multiple_batches() -> Result<()> {
        let mut embedder = create_mock_embedder();
        let mut batch_embedder = BatchEmbedder::new(&mut embedder, 10);

        let texts: Vec<String> = (0..35).map(|i| format!("text {}", i)).collect();
        let embeddings = batch_embedder.embed_batch(&texts)?;

        // should process in 4 batches: 10, 10, 10, 5
        assert_eq!(embeddings.len(), 35);

        Ok(())
    }

    #[test]
    fn test_batch_embedder_batch_size() -> Result<()> {
        let mut embedder = create_mock_embedder();
        let batch_embedder = BatchEmbedder::new(&mut embedder, 16);

        assert_eq!(batch_embedder.batch_size(), 16);

        Ok(())
    }

    #[test]
    fn test_batch_embedder_dimensions() -> Result<()> {
        let mut embedder = create_mock_embedder();
        let batch_embedder = BatchEmbedder::new(&mut embedder, 16);

        assert_eq!(batch_embedder.dimensions(), TEST_DIMENSIONS);

        Ok(())
    }

    #[test]
    fn test_batch_embedder_deterministic() -> Result<()> {
        // verify mock embeddings are deterministic (same text = same embedding)
        let mut embedder = create_mock_embedder();
        let mut batch_embedder = BatchEmbedder::new(&mut embedder, 32);

        let texts = vec!["hello".to_string(), "world".to_string()];
        let embeddings1 = batch_embedder.embed_batch(&texts)?;

        let mut embedder2 = create_mock_embedder();
        let mut batch_embedder2 = BatchEmbedder::new(&mut embedder2, 32);
        let embeddings2 = batch_embedder2.embed_batch(&texts)?;

        assert_eq!(embeddings1, embeddings2);

        Ok(())
    }

    #[tokio::test]
    async fn test_async_batch_embedder() -> Result<()> {
        let embedder = create_mock_embedder();
        let async_embedder = AsyncBatchEmbedder::new(&embedder, 32);

        let texts: Vec<String> = (0..10).map(|i| format!("text {}", i)).collect();
        let embeddings = async_embedder.embed_batch(texts).await?;

        assert_eq!(embeddings.len(), 10);
        assert_eq!(embeddings[0].len(), TEST_DIMENSIONS);

        Ok(())
    }

    #[tokio::test]
    async fn test_async_batch_embedder_empty() -> Result<()> {
        let embedder = create_mock_embedder();
        let async_embedder = AsyncBatchEmbedder::new(&embedder, 32);

        let embeddings = async_embedder.embed_batch(vec![]).await?;
        assert_eq!(embeddings.len(), 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_async_batch_embedder_dimensions() -> Result<()> {
        let embedder = create_mock_embedder();
        let async_embedder = AsyncBatchEmbedder::new(&embedder, 32);

        assert_eq!(async_embedder.dimensions(), TEST_DIMENSIONS);

        Ok(())
    }

    #[tokio::test]
    async fn test_async_batch_embedder_deterministic() -> Result<()> {
        // verify async embeddings are deterministic
        let embedder = create_mock_embedder();
        let async_embedder = AsyncBatchEmbedder::new(&embedder, 32);

        let texts = vec!["hello".to_string(), "world".to_string()];
        let embeddings1 = async_embedder.embed_batch(texts.clone()).await?;
        let embeddings2 = async_embedder.embed_batch(texts).await?;

        assert_eq!(embeddings1, embeddings2);

        Ok(())
    }
}
