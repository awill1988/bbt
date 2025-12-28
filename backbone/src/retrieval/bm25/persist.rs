use crate::error::{BbtError, Result};
use crate::retrieval::bm25::index::Bm25Index;
use std::fs;
use std::path::Path;

/// save bm25 index to binary file using rkyv
///
/// uses atomic write pattern: write to temp file, then rename
pub fn save_index(index: &Bm25Index, path: &Path) -> Result<()> {
    // create temp file path
    let temp_path = path.with_extension("tmp");

    // serialize to bytes
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(index)
        .map_err(|e| BbtError::Schema(format!("failed to serialize bm25 index: {}", e)))?;

    // write to temp file
    fs::write(&temp_path, bytes).map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!("failed to write temp file {:?}: {}", temp_path, e),
        )
    })?;

    // atomic rename
    fs::rename(&temp_path, path).map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!("failed to rename {:?} to {:?}: {}", temp_path, path, e),
        )
    })?;

    tracing::info!(
        "saved bm25 index with {} documents to {}",
        index.num_docs(),
        path.display()
    );

    Ok(())
}

/// load bm25 index from binary file using rkyv (zero-copy deserialization)
pub fn load_index(path: &Path) -> Result<Bm25Index> {
    if !path.exists() {
        tracing::debug!("bm25 index not found at {:?}, creating new index", path);
        return Ok(Bm25Index::new());
    }

    // read file into memory
    let bytes = fs::read(path).map_err(|e| {
        std::io::Error::new(e.kind(), format!("failed to read index file {:?}: {}", path, e))
    })?;

    // deserialize from bytes (zero-copy)
    let index: Bm25Index = rkyv::from_bytes::<Bm25Index, rkyv::rancor::Error>(&bytes)
        .map_err(|e| BbtError::Schema(format!("failed to deserialize bm25 index: {}", e)))?;

    tracing::info!(
        "loaded bm25 index with {} documents from {}",
        index.num_docs(),
        path.display()
    );

    Ok(index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::tempdir;

    #[test]
    fn test_save_load_index() {
        let dir = tempdir().unwrap();
        let index_path = dir.path().join("test_index.bin");

        // create and populate index
        let mut index = Bm25Index::new();
        index.add_document("doc1", "hello world").unwrap();
        index.add_document("doc2", "hello rust").unwrap();

        // save index
        save_index(&index, &index_path).unwrap();
        assert!(index_path.exists());

        // load index
        let loaded = load_index(&index_path).unwrap();
        assert_eq!(loaded.num_docs(), 2);
        assert_eq!(loaded.term_doc_freq("hello"), 2);
        assert_eq!(loaded.term_doc_freq("world"), 1);
    }

    #[test]
    fn test_load_nonexistent_index() {
        let path = PathBuf::from("/tmp/nonexistent_index.bin");
        let index = load_index(&path).unwrap();
        assert_eq!(index.num_docs(), 0);
    }
}
