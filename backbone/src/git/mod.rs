///! git commit indexing and analysis
///!
///! this module provides functionality for extracting, chunking, and analyzing
///! git commit history for semantic search and developer insights

pub mod models;
pub mod extractor;
pub mod chunking;
pub mod classification;
pub mod statistics;

// re-export main types
pub use models::{
    CommitDocument, CommitMetadata, CommitChunk, CommitChunkMetadata,
    CommitClassification, DiffStats, GitSignature, ChunkType,
};
pub use extractor::CommitExtractor;
pub use chunking::{chunk_commit, CommitChunkingStrategy};
pub use classification::classify_commit;
pub use statistics::{
    CommitReference, DeveloperStats, FileEvolution,
    generate_developer_stats, generate_file_evolution,
};
