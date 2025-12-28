use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use uuid::Uuid;

/// git commit document (mirrors Document from document/loader.rs)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitDocument {
    /// unique identifier (uuid v4)
    pub id: String,

    /// git commit sha1 hash (40 chars)
    pub commit_hash: String,

    /// path to git repository
    pub repository_path: PathBuf,

    /// full commit message
    pub message: String,

    /// complete diff as text
    pub diff_text: String,

    /// commit metadata
    pub metadata: CommitMetadata,
}

impl CommitDocument {
    pub fn new(
        commit_hash: String,
        repository_path: PathBuf,
        message: String,
        diff_text: String,
        metadata: CommitMetadata,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            commit_hash,
            repository_path,
            message,
            diff_text,
            metadata,
        }
    }
}

/// commit metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitMetadata {
    pub repository_name: String,
    pub author: GitSignature,
    pub committer: GitSignature,
    pub commit_time: DateTime<Utc>,
    pub parent_hashes: Vec<String>,
    pub classification: CommitClassification,
    pub diff_stats: DiffStats,
    pub custom: HashMap<String, String>,
}

/// git signature (author or committer)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitSignature {
    pub name: String,
    pub email: String,
    pub timestamp: DateTime<Utc>,
}

/// conventional commit classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CommitClassification {
    Feat,
    Fix,
    Refactor,
    Docs,
    Test,
    Chore,
    Style,
    Perf,
    Ci,
    Build,
    Revert,
    Unknown,
}

impl CommitClassification {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Feat => "feat",
            Self::Fix => "fix",
            Self::Refactor => "refactor",
            Self::Docs => "docs",
            Self::Test => "test",
            Self::Chore => "chore",
            Self::Style => "style",
            Self::Perf => "perf",
            Self::Ci => "ci",
            Self::Build => "build",
            Self::Revert => "revert",
            Self::Unknown => "unknown",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "feat" | "feature" => Self::Feat,
            "fix" => Self::Fix,
            "refactor" | "refact" => Self::Refactor,
            "docs" | "doc" => Self::Docs,
            "test" => Self::Test,
            "chore" => Self::Chore,
            "style" => Self::Style,
            "perf" | "performance" => Self::Perf,
            "ci" => Self::Ci,
            "build" => Self::Build,
            "revert" => Self::Revert,
            _ => Self::Unknown,
        }
    }
}

/// diff statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffStats {
    pub files_changed: usize,
    pub insertions: usize,
    pub deletions: usize,
    pub file_paths: Vec<String>,
    pub languages: HashSet<String>,
}

impl DiffStats {
    pub fn new() -> Self {
        Self {
            files_changed: 0,
            insertions: 0,
            deletions: 0,
            file_paths: Vec::new(),
            languages: HashSet::new(),
        }
    }
}

impl Default for DiffStats {
    fn default() -> Self {
        Self::new()
    }
}

/// commit chunk (mirrors Chunk from document/loader.rs)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitChunk {
    /// unique identifier (uuid v4)
    pub id: String,

    /// parent commit document id
    pub commit_document_id: String,

    /// chunk text content
    pub content: String,

    /// approximate token count
    pub tokens: usize,

    /// embedding vector (optional, set later)
    pub embedding: Option<Vec<f32>>,

    /// chunk metadata
    pub metadata: CommitChunkMetadata,
}

impl CommitChunk {
    pub fn new(
        commit_document_id: String,
        content: String,
        tokens: usize,
        metadata: CommitChunkMetadata,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            commit_document_id,
            content,
            tokens,
            embedding: None,
            metadata,
        }
    }

    pub fn set_embedding(&mut self, embedding: Vec<f32>) {
        self.embedding = Some(embedding);
    }
}

/// commit chunk metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitChunkMetadata {
    pub chunk_type: ChunkType,
    pub chunk_index: usize,
    pub total_chunks: usize,
    pub file_path: Option<String>,
    pub language: Option<String>,
}

/// type of commit chunk
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChunkType {
    Message,      // commit message
    DiffHeader,   // file-level diff metadata
    DiffContent,  // actual code changes
}

impl ChunkType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Message => "message",
            Self::DiffHeader => "diff_header",
            Self::DiffContent => "diff_content",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_commit_classification_from_str() {
        assert_eq!(CommitClassification::from_str("feat"), CommitClassification::Feat);
        assert_eq!(CommitClassification::from_str("feature"), CommitClassification::Feat);
        assert_eq!(CommitClassification::from_str("fix"), CommitClassification::Fix);
        assert_eq!(CommitClassification::from_str("REFACTOR"), CommitClassification::Refactor);
        assert_eq!(CommitClassification::from_str("unknown"), CommitClassification::Unknown);
    }

    #[test]
    fn test_commit_classification_as_str() {
        assert_eq!(CommitClassification::Feat.as_str(), "feat");
        assert_eq!(CommitClassification::Fix.as_str(), "fix");
        assert_eq!(CommitClassification::Unknown.as_str(), "unknown");
    }

    #[test]
    fn test_chunk_type_as_str() {
        assert_eq!(ChunkType::Message.as_str(), "message");
        assert_eq!(ChunkType::DiffHeader.as_str(), "diff_header");
        assert_eq!(ChunkType::DiffContent.as_str(), "diff_content");
    }

    #[test]
    fn test_commit_chunk_creation() {
        let metadata = CommitChunkMetadata {
            chunk_type: ChunkType::Message,
            chunk_index: 0,
            total_chunks: 1,
            file_path: None,
            language: None,
        };

        let chunk = CommitChunk::new(
            "commit-123".to_string(),
            "test content".to_string(),
            5,
            metadata,
        );

        assert_eq!(chunk.commit_document_id, "commit-123");
        assert_eq!(chunk.content, "test content");
        assert_eq!(chunk.tokens, 5);
        assert!(chunk.embedding.is_none());
    }

    #[test]
    fn test_commit_chunk_set_embedding() {
        let metadata = CommitChunkMetadata {
            chunk_type: ChunkType::Message,
            chunk_index: 0,
            total_chunks: 1,
            file_path: None,
            language: None,
        };

        let mut chunk = CommitChunk::new(
            "commit-123".to_string(),
            "test content".to_string(),
            5,
            metadata,
        );

        chunk.set_embedding(vec![0.1, 0.2, 0.3]);
        assert!(chunk.embedding.is_some());
        assert_eq!(chunk.embedding.as_ref().unwrap().len(), 3);
    }
}
