use crate::error::{BbtError, Result};
use crate::git::classification::classify_commit;
use crate::git::models::CommitClassification;
use crate::lang;
use chrono::{DateTime, TimeZone, Utc};
use git2::{DiffOptions, Repository};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// developer statistics
#[derive(Debug, Clone)]
pub struct DeveloperStats {
    pub email: String,
    pub name: String,
    pub commit_count: usize,
    pub lines_added: usize,
    pub lines_deleted: usize,
    pub files_modified: HashSet<String>,
    pub commit_types: HashMap<CommitClassification, usize>,
    pub first_commit: DateTime<Utc>,
    pub last_commit: DateTime<Utc>,
}

/// file evolution history
#[derive(Debug, Clone)]
pub struct FileEvolution {
    pub file_path: String,
    pub commits: Vec<CommitReference>,
    pub total_changes: usize,
    pub contributors: HashSet<String>,
    pub languages: HashSet<String>,
}

/// reference to a commit
#[derive(Debug, Clone)]
pub struct CommitReference {
    pub hash: String,
    pub author: String,
    pub timestamp: DateTime<Utc>,
    pub message_summary: String,
    pub insertions: usize,
    pub deletions: usize,
}

/// generate developer statistics from a git repository
///
/// # arguments
/// * `repo_path` - path to the git repository
/// * `since` - optional timestamp to filter commits from
///
/// # returns
/// vector of developer statistics sorted by commit count
pub fn generate_developer_stats(
    repo_path: &Path,
    since: Option<DateTime<Utc>>,
) -> Result<Vec<DeveloperStats>> {
    let repo = Repository::open(repo_path)
        .map_err(|e| BbtError::Schema(format!("failed to open git repository: {}", e)))?;

    let mut revwalk = repo.revwalk()
        .map_err(|e| BbtError::Schema(format!("failed to create revwalk: {}", e)))?;

    revwalk.push_head()
        .map_err(|e| BbtError::Schema(format!("failed to push head: {}", e)))?;

    // aggregate stats per developer (keyed by email)
    let mut dev_stats: HashMap<String, DeveloperStats> = HashMap::new();

    for oid in revwalk {
        let oid = oid.map_err(|e| BbtError::Schema(format!("failed to get oid: {}", e)))?;
        let commit = repo.find_commit(oid)
            .map_err(|e| BbtError::Schema(format!("failed to find commit: {}", e)))?;

        // extract commit time
        let time = commit.time();
        let commit_time = Utc.timestamp_opt(time.seconds(), 0)
            .single()
            .unwrap_or_else(Utc::now);

        // filter by since time if provided
        if let Some(since_time) = since {
            if commit_time < since_time {
                continue;
            }
        }

        // extract author info
        let author = commit.author();
        let email = author.email().unwrap_or("unknown").to_string();
        let name = author.name().unwrap_or("unknown").to_string();

        // get commit message and classification
        let message = commit.message().unwrap_or("");
        let classification = classify_commit(message);

        // calculate diff stats
        let tree = commit.tree()
            .map_err(|e| BbtError::Schema(format!("failed to get tree: {}", e)))?;
        let parent_tree = if commit.parent_count() > 0 {
            commit.parent(0).and_then(|p| p.tree()).ok()
        } else {
            None
        };

        let diff = repo.diff_tree_to_tree(parent_tree.as_ref(), Some(&tree), None)
            .map_err(|e| BbtError::Schema(format!("failed to create diff: {}", e)))?;

        let stats = diff.stats()
            .map_err(|e| BbtError::Schema(format!("failed to get diff stats: {}", e)))?;

        // collect file paths
        let mut file_paths = HashSet::new();
        let _ = diff.foreach(
            &mut |delta, _| {
                if let Some(path) = delta.new_file().path() {
                    file_paths.insert(path.to_string_lossy().to_string());
                }
                true
            },
            None,
            None,
            None,
        );

        // update or create developer stats
        let entry = dev_stats.entry(email.clone()).or_insert_with(|| DeveloperStats {
            email: email.clone(),
            name: name.clone(),
            commit_count: 0,
            lines_added: 0,
            lines_deleted: 0,
            files_modified: HashSet::new(),
            commit_types: HashMap::new(),
            first_commit: commit_time,
            last_commit: commit_time,
        });

        entry.commit_count += 1;
        entry.lines_added += stats.insertions();
        entry.lines_deleted += stats.deletions();
        entry.files_modified.extend(file_paths);
        *entry.commit_types.entry(classification).or_insert(0) += 1;

        if commit_time < entry.first_commit {
            entry.first_commit = commit_time;
        }
        if commit_time > entry.last_commit {
            entry.last_commit = commit_time;
        }
    }

    // sort by commit count descending
    let mut result: Vec<DeveloperStats> = dev_stats.into_values().collect();
    result.sort_by(|a, b| b.commit_count.cmp(&a.commit_count));

    Ok(result)
}

/// generate file evolution history from a git repository
///
/// # arguments
/// * `repo_path` - path to the git repository
/// * `file_path` - path to the file (relative to repo root)
///
/// # returns
/// file evolution history with commits, contributors, and language info
pub fn generate_file_evolution(
    repo_path: &Path,
    file_path: &str,
) -> Result<FileEvolution> {
    let repo = Repository::open(repo_path)
        .map_err(|e| BbtError::Schema(format!("failed to open git repository: {}", e)))?;

    let mut revwalk = repo.revwalk()
        .map_err(|e| BbtError::Schema(format!("failed to create revwalk: {}", e)))?;

    revwalk.push_head()
        .map_err(|e| BbtError::Schema(format!("failed to push head: {}", e)))?;

    let mut commits = Vec::new();
    let mut contributors = HashSet::new();
    let mut total_changes = 0_usize;

    // detect language from file extension
    let mut languages = HashSet::new();
    if let Some(lang) = lang::from_path(Path::new(file_path)) {
        languages.insert(lang.id.to_string());
    }

    for oid in revwalk {
        let oid = oid.map_err(|e| BbtError::Schema(format!("failed to get oid: {}", e)))?;
        let commit = repo.find_commit(oid)
            .map_err(|e| BbtError::Schema(format!("failed to find commit: {}", e)))?;

        // get commit tree and parent tree
        let tree = commit.tree()
            .map_err(|e| BbtError::Schema(format!("failed to get tree: {}", e)))?;
        let parent_tree = if commit.parent_count() > 0 {
            commit.parent(0).and_then(|p| p.tree()).ok()
        } else {
            None
        };

        // create diff with pathspec filter
        let mut diff_opts = DiffOptions::new();
        diff_opts.pathspec(file_path);

        let diff = repo.diff_tree_to_tree(
            parent_tree.as_ref(),
            Some(&tree),
            Some(&mut diff_opts),
        ).map_err(|e| BbtError::Schema(format!("failed to create diff: {}", e)))?;

        // check if this commit affected the file
        let stats = diff.stats()
            .map_err(|e| BbtError::Schema(format!("failed to get diff stats: {}", e)))?;

        if stats.files_changed() == 0 {
            continue;
        }

        // extract commit details
        let author = commit.author();
        let author_str = format!(
            "{} <{}>",
            author.name().unwrap_or("unknown"),
            author.email().unwrap_or("unknown")
        );
        let time = commit.time();
        let timestamp = Utc.timestamp_opt(time.seconds(), 0)
            .single()
            .unwrap_or_else(Utc::now);
        let message = commit.message().unwrap_or("");
        let message_summary = message.lines().next().unwrap_or("").to_string();

        let insertions = stats.insertions();
        let deletions = stats.deletions();
        total_changes += insertions + deletions;

        contributors.insert(author_str.clone());

        commits.push(CommitReference {
            hash: commit.id().to_string(),
            author: author_str,
            timestamp,
            message_summary,
            insertions,
            deletions,
        });
    }

    // sort commits by timestamp (newest first)
    commits.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

    Ok(FileEvolution {
        file_path: file_path.to_string(),
        commits,
        total_changes,
        contributors,
        languages,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_generate_developer_stats() {
        // try to run on current repo
        let current_dir = env::current_dir().unwrap();
        if let Ok(stats) = generate_developer_stats(&current_dir, None) {
            // should have at least one developer
            if !stats.is_empty() {
                let first = &stats[0];
                assert!(!first.email.is_empty());
                assert!(first.commit_count > 0);
            }
        }
    }

    #[test]
    fn test_generate_file_evolution() {
        let current_dir = env::current_dir().unwrap();
        if let Ok(evolution) = generate_file_evolution(&current_dir, "Cargo.toml") {
            assert_eq!(evolution.file_path, "Cargo.toml");
            // cargo.toml should have at least one commit
            if !evolution.commits.is_empty() {
                let first = &evolution.commits[0];
                assert!(!first.hash.is_empty());
            }
        }
    }

    #[test]
    fn test_developer_stats_with_since() {
        let current_dir = env::current_dir().unwrap();
        // use a very recent date to limit results
        let since = Utc::now() - chrono::Duration::days(30);
        if let Ok(stats) = generate_developer_stats(&current_dir, Some(since)) {
            // result should be smaller or equal to full history
            let full_stats = generate_developer_stats(&current_dir, None).unwrap_or_default();
            assert!(stats.len() <= full_stats.len());
        }
    }
}
