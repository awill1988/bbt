use crate::error::Result;
use crate::git::models::{CommitChunk, CommitChunkMetadata, CommitDocument, ChunkType};
use crate::lang;
use serde::{Deserialize, Serialize};

/// commit chunking strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommitChunkingStrategy {
    MessageOnly,       // only commit message
    MessageAndSummary, // message + diff stats (no code)
    ByFile,            // message + one chunk per file (default)
    ByHunk,            // message + chunks per diff hunk
    SlidingWindow,     // message + sliding window over diff
}

/// parsed file diff from git diff output
#[derive(Debug, Clone)]
struct FileDiff {
    file_path: String,
    header: String,
    content: String,
}

/// parsed hunk from git diff output
#[derive(Debug, Clone)]
struct DiffHunk {
    file_path: String,
    header: String,  // @@ -X,Y +A,B @@ context
    content: String, // actual diff lines
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
/// creates one chunk for the message and one chunk per file in the diff
fn chunk_by_file(commit: &CommitDocument, chunk_size: usize) -> Result<Vec<CommitChunk>> {
    let file_diffs = parse_file_diffs(&commit.diff_text);

    // if no file diffs found, fall back to message only
    if file_diffs.is_empty() {
        return chunk_message_only(commit);
    }

    let total_chunks = 1 + file_diffs.len(); // message + files
    let mut chunks = Vec::with_capacity(total_chunks);

    // first chunk: commit message
    chunks.push(CommitChunk::new(
        commit.id.clone(),
        commit.message.clone(),
        estimate_tokens(&commit.message),
        CommitChunkMetadata {
            chunk_type: ChunkType::Message,
            chunk_index: 0,
            total_chunks,
            file_path: None,
            language: None,
        },
    ));

    // one chunk per file
    for (idx, file_diff) in file_diffs.iter().enumerate() {
        let language = lang::from_path(&file_diff.file_path).map(|l| l.id.to_string());

        // format: file path + header + content, truncated to chunk_size tokens
        let content = format!(
            "File: {}\n{}\n{}",
            file_diff.file_path, file_diff.header, file_diff.content
        );
        let truncated = truncate_to_tokens(&content, chunk_size);

        chunks.push(CommitChunk::new(
            commit.id.clone(),
            truncated.clone(),
            estimate_tokens(&truncated),
            CommitChunkMetadata {
                chunk_type: ChunkType::DiffContent,
                chunk_index: idx + 1,
                total_chunks,
                file_path: Some(file_diff.file_path.clone()),
                language,
            },
        ));
    }

    Ok(chunks)
}

/// chunk by diff hunk
/// creates one chunk for the message and one chunk per hunk (@@...@@) in the diff
fn chunk_by_hunk(commit: &CommitDocument, chunk_size: usize) -> Result<Vec<CommitChunk>> {
    let hunks = parse_diff_hunks(&commit.diff_text);

    // if no hunks found, fall back to message only
    if hunks.is_empty() {
        return chunk_message_only(commit);
    }

    let total_chunks = 1 + hunks.len(); // message + hunks
    let mut chunks = Vec::with_capacity(total_chunks);

    // first chunk: commit message
    chunks.push(CommitChunk::new(
        commit.id.clone(),
        commit.message.clone(),
        estimate_tokens(&commit.message),
        CommitChunkMetadata {
            chunk_type: ChunkType::Message,
            chunk_index: 0,
            total_chunks,
            file_path: None,
            language: None,
        },
    ));

    // one chunk per hunk
    for (idx, hunk) in hunks.iter().enumerate() {
        let language = lang::from_path(&hunk.file_path).map(|l| l.id.to_string());

        // format: file path + hunk header + content
        let content = format!(
            "File: {}\n{}\n{}",
            hunk.file_path, hunk.header, hunk.content
        );
        let truncated = truncate_to_tokens(&content, chunk_size);

        chunks.push(CommitChunk::new(
            commit.id.clone(),
            truncated.clone(),
            estimate_tokens(&truncated),
            CommitChunkMetadata {
                chunk_type: ChunkType::DiffContent,
                chunk_index: idx + 1,
                total_chunks,
                file_path: Some(hunk.file_path.clone()),
                language,
            },
        ));
    }

    Ok(chunks)
}

/// chunk by sliding window over diff
/// creates one chunk for the message and sliding window chunks over the diff text
fn chunk_by_sliding_window(
    commit: &CommitDocument,
    chunk_size: usize,
    overlap: usize,
) -> Result<Vec<CommitChunk>> {
    // if diff is empty, fall back to message only
    if commit.diff_text.trim().is_empty() {
        return chunk_message_only(commit);
    }

    // create windows over the diff text
    let windows = create_sliding_windows(&commit.diff_text, chunk_size, overlap);

    let total_chunks = 1 + windows.len(); // message + windows
    let mut chunks = Vec::with_capacity(total_chunks);

    // first chunk: commit message
    chunks.push(CommitChunk::new(
        commit.id.clone(),
        commit.message.clone(),
        estimate_tokens(&commit.message),
        CommitChunkMetadata {
            chunk_type: ChunkType::Message,
            chunk_index: 0,
            total_chunks,
            file_path: None,
            language: None,
        },
    ));

    // sliding window chunks
    for (idx, window) in windows.iter().enumerate() {
        chunks.push(CommitChunk::new(
            commit.id.clone(),
            window.clone(),
            estimate_tokens(window),
            CommitChunkMetadata {
                chunk_type: ChunkType::DiffContent,
                chunk_index: idx + 1,
                total_chunks,
                file_path: None,
                language: None,
            },
        ));
    }

    Ok(chunks)
}

/// estimate token count from text (simple word count approximation)
/// uses ~1.3 tokens per word as approximation for code
fn estimate_tokens(text: &str) -> usize {
    let words = text.split_whitespace().count();
    (words as f32 * 1.3) as usize
}

/// truncate text to approximately the given number of tokens
fn truncate_to_tokens(text: &str, max_tokens: usize) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    // approximate: max_tokens / 1.3 words
    let max_words = (max_tokens as f32 / 1.3) as usize;

    if words.len() <= max_words {
        text.to_string()
    } else {
        words[..max_words].join(" ") + "..."
    }
}

/// parse git diff output into per-file diffs
fn parse_file_diffs(diff_text: &str) -> Vec<FileDiff> {
    let mut file_diffs = Vec::new();
    let mut current_file: Option<FileDiff> = None;
    let mut in_header = false;

    for line in diff_text.lines() {
        if line.starts_with("diff --git ") {
            // save previous file diff if any
            if let Some(fd) = current_file.take() {
                file_diffs.push(fd);
            }

            // extract file path from "diff --git a/path b/path"
            let file_path = extract_file_path_from_diff_line(line);
            current_file = Some(FileDiff {
                file_path,
                header: line.to_string(),
                content: String::new(),
            });
            in_header = true;
        } else if let Some(ref mut fd) = current_file {
            if in_header {
                // header lines: index, ---, +++, etc.
                if line.starts_with("@@") {
                    in_header = false;
                    fd.content.push_str(line);
                    fd.content.push('\n');
                } else {
                    fd.header.push('\n');
                    fd.header.push_str(line);
                }
            } else {
                fd.content.push_str(line);
                fd.content.push('\n');
            }
        }
    }

    // save last file diff
    if let Some(fd) = current_file {
        file_diffs.push(fd);
    }

    file_diffs
}

/// parse git diff output into hunks
fn parse_diff_hunks(diff_text: &str) -> Vec<DiffHunk> {
    let mut hunks = Vec::new();
    let mut current_file_path = String::new();
    let mut current_hunk: Option<DiffHunk> = None;

    for line in diff_text.lines() {
        if line.starts_with("diff --git ") {
            // save previous hunk if any
            if let Some(hunk) = current_hunk.take() {
                hunks.push(hunk);
            }
            current_file_path = extract_file_path_from_diff_line(line);
        } else if line.starts_with("@@") {
            // save previous hunk if any
            if let Some(hunk) = current_hunk.take() {
                hunks.push(hunk);
            }

            // start new hunk
            current_hunk = Some(DiffHunk {
                file_path: current_file_path.clone(),
                header: line.to_string(),
                content: String::new(),
            });
        } else if let Some(ref mut hunk) = current_hunk {
            // only include actual diff lines (+, -, space, or context)
            if line.starts_with('+')
                || line.starts_with('-')
                || line.starts_with(' ')
                || line.starts_with('\\')
            {
                hunk.content.push_str(line);
                hunk.content.push('\n');
            }
        }
    }

    // save last hunk
    if let Some(hunk) = current_hunk {
        hunks.push(hunk);
    }

    hunks
}

/// extract file path from "diff --git a/path b/path" line
fn extract_file_path_from_diff_line(line: &str) -> String {
    // format: "diff --git a/path/to/file b/path/to/file"
    if let Some(b_part) = line.split(" b/").nth(1) {
        b_part.to_string()
    } else if let Some(a_part) = line.split(" a/").nth(1) {
        // fallback: use a/ path
        a_part.split_whitespace().next().unwrap_or("unknown").to_string()
    } else {
        "unknown".to_string()
    }
}

/// create sliding windows over text
fn create_sliding_windows(text: &str, window_size: usize, overlap: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();

    if words.is_empty() {
        return vec![];
    }

    // convert token counts to word counts (approximate)
    let window_words = (window_size as f32 / 1.3) as usize;
    let overlap_words = (overlap as f32 / 1.3) as usize;
    let step = window_words.saturating_sub(overlap_words).max(1);

    let mut windows = Vec::new();
    let mut start = 0;

    while start < words.len() {
        let end = (start + window_words).min(words.len());
        let window = words[start..end].join(" ");
        windows.push(window);

        start += step;

        // avoid tiny final windows
        if start < words.len() && words.len() - start < step / 2 {
            break;
        }
    }

    windows
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

    fn create_commit_with_diff() -> CommitDocument {
        let diff_text = r#"diff --git a/src/main.rs b/src/main.rs
index 1234567..abcdefg 100644
--- a/src/main.rs
+++ b/src/main.rs
@@ -1,5 +1,6 @@
 fn main() {
-    println!("Hello, world!");
+    println!("Hello, Rust!");
+    println!("Welcome!");
 }
diff --git a/src/lib.rs b/src/lib.rs
index 2345678..bcdefgh 100644
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -10,3 +10,7 @@ pub fn add(a: i32, b: i32) -> i32 {
     a + b
 }
+
+pub fn subtract(a: i32, b: i32) -> i32 {
+    a - b
+}
"#;
        CommitDocument::new(
            "def456".to_string(),
            PathBuf::from("/test/repo"),
            "feat: update greeting and add subtract function".to_string(),
            diff_text.to_string(),
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
                    insertions: 5,
                    deletions: 1,
                    file_paths: vec!["src/main.rs".to_string(), "src/lib.rs".to_string()],
                    languages: HashSet::new(),
                },
                custom: HashMap::new(),
            },
        )
    }

    #[test]
    fn test_chunk_by_file() {
        let commit = create_commit_with_diff();
        let chunks = chunk_commit(&commit, CommitChunkingStrategy::ByFile, 512, 128).unwrap();

        // should have 3 chunks: message + 2 files
        assert_eq!(chunks.len(), 3);

        // first chunk is message
        assert_eq!(chunks[0].metadata.chunk_type, ChunkType::Message);
        assert!(chunks[0].content.contains("update greeting"));

        // second chunk is first file
        assert_eq!(chunks[1].metadata.chunk_type, ChunkType::DiffContent);
        assert_eq!(chunks[1].metadata.file_path, Some("src/main.rs".to_string()));
        assert_eq!(chunks[1].metadata.language, Some("rust".to_string()));
        assert!(chunks[1].content.contains("Hello, Rust!"));

        // third chunk is second file
        assert_eq!(chunks[2].metadata.chunk_type, ChunkType::DiffContent);
        assert_eq!(chunks[2].metadata.file_path, Some("src/lib.rs".to_string()));
        assert!(chunks[2].content.contains("subtract"));
    }

    #[test]
    fn test_chunk_by_hunk() {
        let commit = create_commit_with_diff();
        let chunks = chunk_commit(&commit, CommitChunkingStrategy::ByHunk, 512, 128).unwrap();

        // should have 3 chunks: message + 2 hunks
        assert_eq!(chunks.len(), 3);

        // first chunk is message
        assert_eq!(chunks[0].metadata.chunk_type, ChunkType::Message);

        // hunks should have @@ headers
        assert!(chunks[1].content.contains("@@"));
        assert!(chunks[2].content.contains("@@"));
    }

    #[test]
    fn test_chunk_by_sliding_window() {
        let commit = create_commit_with_diff();
        let chunks = chunk_commit(&commit, CommitChunkingStrategy::SlidingWindow, 50, 10).unwrap();

        // should have message + multiple window chunks
        assert!(chunks.len() >= 2);

        // first chunk is always message
        assert_eq!(chunks[0].metadata.chunk_type, ChunkType::Message);

        // rest are diff content windows
        for chunk in &chunks[1..] {
            assert_eq!(chunk.metadata.chunk_type, ChunkType::DiffContent);
        }
    }

    #[test]
    fn test_chunk_by_file_empty_diff() {
        let mut commit = create_test_commit();
        commit.diff_text = String::new();
        let chunks = chunk_commit(&commit, CommitChunkingStrategy::ByFile, 512, 128).unwrap();

        // should fall back to message only
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].metadata.chunk_type, ChunkType::Message);
    }

    #[test]
    fn test_parse_file_diffs() {
        let diff = r#"diff --git a/file1.rs b/file1.rs
index 123..456 100644
--- a/file1.rs
+++ b/file1.rs
@@ -1,3 +1,4 @@
+new line
 old line
diff --git a/file2.py b/file2.py
index 789..abc 100644
--- a/file2.py
+++ b/file2.py
@@ -1 +1 @@
-old
+new
"#;
        let file_diffs = parse_file_diffs(diff);

        assert_eq!(file_diffs.len(), 2);
        assert_eq!(file_diffs[0].file_path, "file1.rs");
        assert_eq!(file_diffs[1].file_path, "file2.py");
    }

    #[test]
    fn test_parse_diff_hunks() {
        let diff = r#"diff --git a/file.rs b/file.rs
index 123..456 100644
--- a/file.rs
+++ b/file.rs
@@ -1,3 +1,4 @@ fn first()
+new line 1
 context
@@ -10,2 +11,3 @@ fn second()
+new line 2
 more context
"#;
        let hunks = parse_diff_hunks(diff);

        assert_eq!(hunks.len(), 2);
        assert_eq!(hunks[0].file_path, "file.rs");
        assert!(hunks[0].header.contains("-1,3 +1,4"));
        assert!(hunks[1].header.contains("-10,2 +11,3"));
    }

    #[test]
    fn test_language_detection() {
        // use the centralized lang module
        assert_eq!(lang::from_path("src/main.rs").map(|l| l.id), Some("rust"));
        assert_eq!(lang::from_path("app.py").map(|l| l.id), Some("python"));
        assert_eq!(lang::from_path("index.ts").map(|l| l.id), Some("typescript"));
        assert_eq!(lang::from_path("styles.css").map(|l| l.id), Some("css"));
        assert!(lang::from_path("no_extension").is_none());
        assert!(lang::from_path("unknown.xyz").is_none());
    }

    #[test]
    fn test_extract_file_path() {
        assert_eq!(
            extract_file_path_from_diff_line("diff --git a/src/main.rs b/src/main.rs"),
            "src/main.rs"
        );
        assert_eq!(
            extract_file_path_from_diff_line("diff --git a/old.txt b/new.txt"),
            "new.txt"
        );
    }

    #[test]
    fn test_sliding_windows() {
        let text = "one two three four five six seven eight nine ten";
        // window of ~4 tokens (3 words), overlap ~1 token
        let windows = create_sliding_windows(text, 4, 1);

        assert!(!windows.is_empty());
        // windows should overlap
        if windows.len() > 1 {
            // check that windows are different
            assert_ne!(windows[0], windows[1]);
        }
    }

    #[test]
    fn test_truncate_to_tokens() {
        let text = "one two three four five six seven eight nine ten";
        let truncated = truncate_to_tokens(text, 5); // ~3-4 words
        assert!(truncated.len() < text.len());
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn test_estimate_tokens() {
        let text = "hello world foo bar";
        let tokens = estimate_tokens(text);
        // 4 words * 1.3 ≈ 5 tokens
        assert_eq!(tokens, 5);
    }
}
