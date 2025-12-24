use crate::error::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Processing status for a document
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ProcessingStatus {
    Pending,
    Processing,
    Completed,
    Failed,
}

/// State information for a processed document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingState {
    /// SHA256 hash of the file
    pub file_hash: String,

    /// Source file path
    pub source_path: String,

    /// When the file was processed
    pub processed_at: DateTime<Utc>,

    /// Number of chunks created
    pub chunk_count: usize,

    /// Processing status
    pub status: ProcessingStatus,

    /// Error message if failed
    pub error: Option<String>,
}

/// State store for tracking document processing
pub struct StateStore {
    path: PathBuf,
    state: HashMap<String, ProcessingState>,
}

impl StateStore {
    /// Create or load a state store from the given path
    ///
    /// # Arguments
    /// * `path` - Path to the state JSON file
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        let state = if path.exists() {
            let content = fs::read_to_string(&path)?;
            // handle empty file
            if content.trim().is_empty() {
                HashMap::new()
            } else {
                serde_json::from_str(&content)?
            }
        } else {
            HashMap::new()
        };

        Ok(Self { path, state })
    }

    /// Check if a file hash has been processed
    pub fn is_processed(&self, file_hash: &str) -> bool {
        self.state
            .get(file_hash)
            .map(|s| s.status == ProcessingStatus::Completed)
            .unwrap_or(false)
    }

    /// Get processing state for a file hash
    pub fn get(&self, file_hash: &str) -> Option<&ProcessingState> {
        self.state.get(file_hash)
    }

    /// Mark a file as currently being processed
    pub fn mark_processing(&mut self, file_hash: String, source_path: String) -> Result<()> {
        let state = ProcessingState {
            file_hash: file_hash.clone(),
            source_path,
            processed_at: Utc::now(),
            chunk_count: 0,
            status: ProcessingStatus::Processing,
            error: None,
        };
        self.state.insert(file_hash, state);
        self.save()
    }

    /// Mark a file as completed
    pub fn mark_complete(
        &mut self,
        file_hash: &str,
        chunk_count: usize,
    ) -> Result<()> {
        if let Some(state) = self.state.get_mut(file_hash) {
            state.status = ProcessingStatus::Completed;
            state.chunk_count = chunk_count;
            state.processed_at = Utc::now();
            state.error = None;
        }
        self.save()
    }

    /// Mark a file as failed
    pub fn mark_failed(&mut self, file_hash: &str, error: String) -> Result<()> {
        if let Some(state) = self.state.get_mut(file_hash) {
            state.status = ProcessingStatus::Failed;
            state.error = Some(error);
            state.processed_at = Utc::now();
        }
        self.save()
    }

    /// Remove a file from the state
    pub fn remove(&mut self, file_hash: &str) -> Result<()> {
        self.state.remove(file_hash);
        self.save()
    }

    /// Get all file hashes in the state
    pub fn all_hashes(&self) -> Vec<&String> {
        self.state.keys().collect()
    }

    /// Get count of processed files
    pub fn count_completed(&self) -> usize {
        self.state
            .values()
            .filter(|s| s.status == ProcessingStatus::Completed)
            .count()
    }

    /// Save state to disk
    fn save(&self) -> Result<()> {
        // ensure parent directory exists
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }

        let json = serde_json::to_string_pretty(&self.state)?;
        fs::write(&self.path, json)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_state_store_new() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let store = StateStore::new(temp_file.path())?;
        assert_eq!(store.state.len(), 0);
        Ok(())
    }

    #[test]
    fn test_state_store_mark_processing() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let mut store = StateStore::new(temp_file.path())?;

        store.mark_processing("hash123".to_string(), "/path/to/file".to_string())?;

        let state = store.get("hash123").unwrap();
        assert_eq!(state.file_hash, "hash123");
        assert_eq!(state.source_path, "/path/to/file");
        assert_eq!(state.status, ProcessingStatus::Processing);
        assert_eq!(state.chunk_count, 0);

        Ok(())
    }

    #[test]
    fn test_state_store_mark_complete() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let mut store = StateStore::new(temp_file.path())?;

        store.mark_processing("hash123".to_string(), "/path/to/file".to_string())?;
        store.mark_complete("hash123", 42)?;

        let state = store.get("hash123").unwrap();
        assert_eq!(state.status, ProcessingStatus::Completed);
        assert_eq!(state.chunk_count, 42);
        assert!(state.error.is_none());

        Ok(())
    }

    #[test]
    fn test_state_store_mark_failed() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let mut store = StateStore::new(temp_file.path())?;

        store.mark_processing("hash123".to_string(), "/path/to/file".to_string())?;
        store.mark_failed("hash123", "test error".to_string())?;

        let state = store.get("hash123").unwrap();
        assert_eq!(state.status, ProcessingStatus::Failed);
        assert_eq!(state.error, Some("test error".to_string()));

        Ok(())
    }

    #[test]
    fn test_state_store_is_processed() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let mut store = StateStore::new(temp_file.path())?;

        assert!(!store.is_processed("hash123"));

        store.mark_processing("hash123".to_string(), "/path/to/file".to_string())?;
        assert!(!store.is_processed("hash123"));

        store.mark_complete("hash123", 10)?;
        assert!(store.is_processed("hash123"));

        Ok(())
    }

    #[test]
    fn test_state_store_remove() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let mut store = StateStore::new(temp_file.path())?;

        store.mark_processing("hash123".to_string(), "/path/to/file".to_string())?;
        assert!(store.get("hash123").is_some());

        store.remove("hash123")?;
        assert!(store.get("hash123").is_none());

        Ok(())
    }

    #[test]
    fn test_state_store_persistence() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let path = temp_file.path().to_path_buf();

        // create store and add state
        {
            let mut store = StateStore::new(&path)?;
            store.mark_processing("hash123".to_string(), "/path/to/file".to_string())?;
            store.mark_complete("hash123", 10)?;
        }

        // load store from disk
        {
            let store = StateStore::new(&path)?;
            assert!(store.is_processed("hash123"));
            let state = store.get("hash123").unwrap();
            assert_eq!(state.chunk_count, 10);
        }

        Ok(())
    }

    #[test]
    fn test_state_store_count_completed() -> Result<()> {
        let temp_file = NamedTempFile::new()?;
        let mut store = StateStore::new(temp_file.path())?;

        assert_eq!(store.count_completed(), 0);

        store.mark_processing("hash1".to_string(), "/file1".to_string())?;
        store.mark_complete("hash1", 5)?;
        assert_eq!(store.count_completed(), 1);

        store.mark_processing("hash2".to_string(), "/file2".to_string())?;
        assert_eq!(store.count_completed(), 1);

        store.mark_complete("hash2", 3)?;
        assert_eq!(store.count_completed(), 2);

        Ok(())
    }
}
