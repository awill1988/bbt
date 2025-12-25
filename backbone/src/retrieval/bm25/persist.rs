use crate::error::{BbtError, Result};
use crate::retrieval::bm25::index::Bm25Index;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// serialized document entry for jsonl format
#[derive(Debug, Serialize, Deserialize)]
struct DocumentEntry {
    doc_id: String,
    tokens: Vec<String>,
    length: usize,
    chunk_text: String,
}

/// save bm25 index to jsonl file
///
/// uses atomic write pattern: write to temp file, then rename
pub fn save_index(index: &Bm25Index, path: &Path) -> Result<()> {
    // create temp file path
    let temp_path = path.with_extension("tmp");

    // write to temp file
    {
        let file = File::create(&temp_path).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("failed to create temp file {:?}: {}", temp_path, e),
            )
        })?;
        let mut writer = BufWriter::new(file);

        // write each document as jsonl entry
        for doc_id in index.doc_ids() {
            if let Some(stats) = index.get_document(&doc_id) {
                // collect tokens from doc
                let tokens = tokenize(&stats.chunk_text);

                let entry = DocumentEntry {
                    doc_id: doc_id.clone(),
                    tokens,
                    length: stats.doc_length,
                    chunk_text: stats.chunk_text.clone(),
                };

                let json = serde_json::to_string(&entry)?;
                writeln!(writer, "{}", json).map_err(|e| {
                    std::io::Error::new(e.kind(), format!("failed to write entry: {}", e))
                })?;
            }
        }

        writer.flush().map_err(|e| {
            std::io::Error::new(e.kind(), format!("failed to flush writer: {}", e))
        })?;
    }

    // atomic rename
    std::fs::rename(&temp_path, path).map_err(|e| {
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

/// load bm25 index from jsonl file
pub fn load_index(path: &Path) -> Result<Bm25Index> {
    if !path.exists() {
        return Ok(Bm25Index::new());
    }

    let file = File::open(path).map_err(|e| {
        std::io::Error::new(e.kind(), format!("failed to open index file {:?}: {}", path, e))
    })?;
    let reader = BufReader::new(file);

    let mut index = Bm25Index::new();
    let mut line_num = 0;

    for line in reader.lines() {
        line_num += 1;
        let line = line.map_err(|e| {
            std::io::Error::new(e.kind(), format!("failed to read line {}: {}", line_num, e))
        })?;

        if line.trim().is_empty() {
            continue;
        }

        let entry: DocumentEntry = serde_json::from_str(&line).map_err(|e| {
            BbtError::Schema(format!("failed to parse line {}: {}", line_num, e))
        })?;

        // add document to index
        index.add_document(&entry.doc_id, &entry.chunk_text)?;
    }

    tracing::info!(
        "loaded bm25 index with {} documents from {}",
        index.num_docs(),
        path.display()
    );

    Ok(index)
}

/// tokenize text (same as index/scorer)
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .filter(|s| s.len() > 1)
        .map(String::from)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::tempdir;

    #[test]
    fn test_save_load_index() {
        let dir = tempdir().unwrap();
        let index_path = dir.path().join("test_index.jsonl");

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
        let path = PathBuf::from("/tmp/nonexistent_index.jsonl");
        let index = load_index(&path).unwrap();
        assert_eq!(index.num_docs(), 0);
    }
}
