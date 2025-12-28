use crate::error::Result;
use crate::git::models::{CommitDocument, CommitChunk, CommitChunkMetadata, ChunkType};
use serde::{Deserialize, Serialize};

/// commit chunking strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommitChunkingStrategy {
    MessageOnly,        // only commit message
    MessageAndSummary,  // message + diff stats (no code)
    ByFile,            // message + one chunk per file (default)
    ByHunk,            // message + chunks per diff hunk
    SlidingWindow,     // message + sliding window over diff
}

impl Default for CommitChunkingStrategy {
    fn default() -> Self {
        Self::ByFile
    }
}

/// chunk a commit document
pub fn chunk_commit(
    commit: &CommitDocument,
    strategy: CommitChunkingStrategy,
    chunk_size: usize,
    overlap: usize,
) -> Result<Vec<CommitChunk>> {
    match strategy {
        CommitChunkingStrategy::MessageOnly => chunk_message_only(commit),
        CommitChunkingStrategy::MessageAndSummary => chunk_message_and_summary(commit),
        CommitChunkingStrategy::ByFile => chunk_by_file(commit, chunk_size),
        CommitChunkingStrategy::ByHunk => chunk_by_hunk(commit, chunk_size),
        CommitChunkingStrategy::SlidingWindow => chunk_by_sliding_window(commit, chunk_size, overlap),
    }
}

/// chunk only the commit message
fn chunk_message_only(commit: &CommitDocument) -> Result<Vec<CommitChunk>> {
    let tokens = estimate_tokens(&commit.message);

    let chunk = CommitChunk::new(
        commit.id.clone(),
        commit.message.clone(),
        tokens,
        CommitChunkMetadata {
            chunk_type: ChunkType::Message,
            chunk_index: 0,
            total_chunks: 1,
            file_path: None,
            language: None,
        },
    );

    Ok(vec![chunk])
}

/// chunk message + diff summary
fn chunk_message_and_summary(commit: &CommitDocument) -> Result<Vec<CommitChunk>> {
    let summary = format!(
        "{}\n\nFiles changed: {}, Insertions: {}, Deletions: {}",
        commit.message,
        commit.metadata.diff_stats.files_changed,
        commit.metadata.diff_stats.insertions,
        commit.metadata.diff_stats.deletions,
    );

    let tokens = estimate_tokens(&summary);

    let chunk = CommitChunk::new(
        commit.id.clone(),
        summary,
        tokens,
        CommitChunkMetadata {
            chunk_type: ChunkType::Message,
            chunk_index: 0,
            total_chunks: 1,
            file_path: None,
            language: None,
        },
    );

    Ok(vec![chunk])
}

/// chunk by file (default strategy)
fn chunk_by_file(_commit: &CommitDocument, _chunk_size: usize) -> Result<Vec<CommitChunk>> {
    // todo: implement file-based chunking
    // for now, fallback to message only
    chunk_message_only(_commit)
}

/// chunk by diff hunk
fn chunk_by_hunk(_commit: &CommitDocument, _chunk_size: usize) -> Result<Vec<CommitChunk>> {
    // todo: implement hunk-based chunking
    // for now, fallback to message only
    chunk_message_only(_commit)
}

/// chunk by sliding window over diff
fn chunk_by_sliding_window(_commit: &CommitDocument, _chunk_size: usize, _overlap: usize) -> Result<Vec<CommitChunk>> {
    // todo: implement sliding window chunking
    // for now, fallback to message only
    chunk_message_only(_commit)
}

/// estimate token count from text (simple word count approximation)
fn estimate_tokens(text: &str) -> usize {
    text.split_whitespace().count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::git::models::{CommitMetadata, DiffStats, GitSignature, CommitClassification};
    use chrono::Utc;
    use std::collections::{HashMap, HashSet};
    use std::path::PathBuf;

    fn create_test_commit() -> CommitDocument {
        CommitDocument::new(
            "abc123".to_string(),
            PathBuf::from("/test/repo"),
            "feat: add new feature\n\nThis is a test commit.".to_string(),
            "diff content here".to_string(),
            CommitMetadata {
                repository_name: "test-repo".to_string(),
                author: GitSignature {
                    name: "Test User".to_string(),
                    email: "test@example.com".to_string(),
                    timestamp: Utc::now(),
                },
                committer: GitSignature {
                    name: "Test User".to_string(),
                    email: "test@example.com".to_string(),
                    timestamp: Utc::now(),
                },
                commit_time: Utc::now(),
                parent_hashes: vec![],
                classification: CommitClassification::Feat,
                diff_stats: DiffStats {
                    files_changed: 2,
                    insertions: 10,
                    deletions: 5,
                    file_paths: vec!["file1.rs".to_string(), "file2.rs".to_string()],
                    languages: HashSet::new(),
                },
                custom: HashMap::new(),
            },
        )
    }

    #[test]
    fn test_chunk_message_only() {
        let commit = create_test_commit();
        let chunks = chunk_commit(&commit, CommitChunkingStrategy::MessageOnly, 512, 128).unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].metadata.chunk_type, ChunkType::Message);
        assert!(chunks[0].content.contains("feat: add new feature"));
    }

    #[test]
    fn test_chunk_message_and_summary() {
        let commit = create_test_commit();
        let chunks = chunk_commit(&commit, CommitChunkingStrategy::MessageAndSummary, 512, 128).unwrap();

        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("Files changed: 2"));
        assert!(chunks[0].content.contains("Insertions: 10"));
        assert!(chunks[0].content.contains("Deletions: 5"));
    }
}
