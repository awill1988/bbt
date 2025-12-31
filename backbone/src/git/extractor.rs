use crate::error::{BbtError, Result};
use crate::git::models::{CommitDocument, CommitMetadata, DiffStats, GitSignature};
use crate::git::classification::classify_commit;
use crate::lang;
use chrono::{DateTime, TimeZone, Utc};
use git2::{Commit, DiffOptions, Repository};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// extracts commit data from git repositories
pub struct CommitExtractor {
    repo: Repository,
    repo_path: PathBuf,
    repo_name: String,
}

impl CommitExtractor {
    /// create a new commit extractor for a git repository
    pub fn new<P: AsRef<Path>>(repo_path: P) -> Result<Self> {
        let repo_path = repo_path.as_ref();

        // open git repository
        let repo = Repository::open(repo_path)
            .map_err(|e| BbtError::Schema(format!("failed to open git repository at {:?}: {}", repo_path, e)))?;

        // extract repository name from path
        let repo_name = repo_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        Ok(Self {
            repo,
            repo_path: repo_path.to_path_buf(),
            repo_name,
        })
    }

    /// extract commits from head back to specified limit
    pub fn extract_commits(
        &self,
        since: Option<DateTime<Utc>>,
        max_commits: Option<usize>,
    ) -> Result<Vec<CommitDocument>> {
        let mut revwalk = self.repo.revwalk()
            .map_err(|e| BbtError::Schema(format!("failed to create revwalk: {}", e)))?;

        // start from head
        revwalk.push_head()
            .map_err(|e| BbtError::Schema(format!("failed to push head: {}", e)))?;

        let mut commits = Vec::new();
        let max = max_commits.unwrap_or(usize::MAX);

        for oid in revwalk {
            if commits.len() >= max {
                break;
            }

            let oid = oid.map_err(|e| BbtError::Schema(format!("failed to get oid: {}", e)))?;
            let commit = self.repo.find_commit(oid)
                .map_err(|e| BbtError::Schema(format!("failed to find commit {}: {}", oid, e)))?;

            // check time filter
            if let Some(since_time) = since {
                let commit_time = self.extract_commit_time(&commit);
                if commit_time < since_time {
                    continue;
                }
            }

            // extract commit document
            match self.extract_commit(&commit) {
                Ok(doc) => commits.push(doc),
                Err(e) => {
                    tracing::warn!("failed to extract commit {}: {}", commit.id(), e);
                    continue;
                }
            }
        }

        Ok(commits)
    }

    /// extract a single commit by hash
    pub fn extract_commit_by_hash(&self, commit_hash: &str) -> Result<CommitDocument> {
        let oid = git2::Oid::from_str(commit_hash)
            .map_err(|e| BbtError::Schema(format!("invalid commit hash {}: {}", commit_hash, e)))?;

        let commit = self.repo.find_commit(oid)
            .map_err(|e| BbtError::Schema(format!("failed to find commit {}: {}", commit_hash, e)))?;

        self.extract_commit(&commit)
    }

    /// extract commit document from git2 commit
    fn extract_commit(&self, commit: &Commit) -> Result<CommitDocument> {
        let commit_hash = commit.id().to_string();
        let message = commit.message().unwrap_or("").to_string();

        // generate diff
        let diff_text = self.generate_diff(commit)?;

        // calculate diff stats
        let diff_stats = self.calculate_diff_stats(commit)?;

        // extract author and committer
        let author = self.extract_signature(&commit.author());
        let committer = self.extract_signature(&commit.committer());

        // extract parent hashes
        let parent_hashes = commit.parent_ids()
            .map(|id| id.to_string())
            .collect();

        // classify commit
        let classification = classify_commit(&message);

        let metadata = CommitMetadata {
            repository_name: self.repo_name.clone(),
            author,
            committer,
            commit_time: self.extract_commit_time(commit),
            parent_hashes,
            classification,
            diff_stats,
            custom: HashMap::new(),
        };

        Ok(CommitDocument::new(
            commit_hash,
            self.repo_path.clone(),
            message,
            diff_text,
            metadata,
        ))
    }

    /// generate full diff text for a commit
    fn generate_diff(&self, commit: &Commit) -> Result<String> {
        let tree = commit.tree()
            .map_err(|e| BbtError::Schema(format!("failed to get commit tree: {}", e)))?;

        let parent_tree = if commit.parent_count() > 0 {
            Some(commit.parent(0)
                .and_then(|p| p.tree())
                .map_err(|e| BbtError::Schema(format!("failed to get parent tree: {}", e)))?)
        } else {
            None
        };

        let mut diff_opts = DiffOptions::new();
        diff_opts.context_lines(3);
        diff_opts.interhunk_lines(0);

        let diff = self.repo.diff_tree_to_tree(
            parent_tree.as_ref(),
            Some(&tree),
            Some(&mut diff_opts),
        ).map_err(|e| BbtError::Schema(format!("failed to create diff: {}", e)))?;

        // convert diff to string
        let mut diff_text = String::new();
        diff.print(git2::DiffFormat::Patch, |_delta, _hunk, line| {
            let content = String::from_utf8_lossy(line.content());
            diff_text.push_str(&content);
            true
        }).map_err(|e| BbtError::Schema(format!("failed to print diff: {}", e)))?;

        Ok(diff_text)
    }

    /// calculate diff statistics
    fn calculate_diff_stats(&self, commit: &Commit) -> Result<DiffStats> {
        let tree = commit.tree()
            .map_err(|e| BbtError::Schema(format!("failed to get commit tree: {}", e)))?;

        let parent_tree = if commit.parent_count() > 0 {
            Some(commit.parent(0)
                .and_then(|p| p.tree())
                .map_err(|e| BbtError::Schema(format!("failed to get parent tree: {}", e)))?)
        } else {
            None
        };

        let diff = self.repo.diff_tree_to_tree(
            parent_tree.as_ref(),
            Some(&tree),
            None,
        ).map_err(|e| BbtError::Schema(format!("failed to create diff: {}", e)))?;

        let stats = diff.stats()
            .map_err(|e| BbtError::Schema(format!("failed to get diff stats: {}", e)))?;

        let mut file_paths = Vec::new();
        let mut languages = HashSet::new();

        diff.foreach(
            &mut |delta, _progress| {
                if let Some(path) = delta.new_file().path() {
                    let path_str = path.to_string_lossy().to_string();
                    file_paths.push(path_str.clone());

                    // detect language using centralized lang module
                    if let Some(language) = lang::from_path(path) {
                        languages.insert(language.id.to_string());
                    }
                }
                true
            },
            None,
            None,
            None,
        ).map_err(|e| BbtError::Schema(format!("failed to foreach diff: {}", e)))?;

        Ok(DiffStats {
            files_changed: stats.files_changed(),
            insertions: stats.insertions(),
            deletions: stats.deletions(),
            file_paths,
            languages,
        })
    }

    /// extract git signature as GitSignature
    fn extract_signature(&self, sig: &git2::Signature) -> GitSignature {
        let name = sig.name().unwrap_or("unknown").to_string();
        let email = sig.email().unwrap_or("unknown").to_string();
        let timestamp = Utc.timestamp_opt(sig.when().seconds(), 0)
            .single()
            .unwrap_or_else(|| Utc::now());

        GitSignature {
            name,
            email,
            timestamp,
        }
    }

    /// extract commit time
    fn extract_commit_time(&self, commit: &Commit) -> DateTime<Utc> {
        let time = commit.time();
        Utc.timestamp_opt(time.seconds(), 0)
            .single()
            .unwrap_or_else(|| Utc::now())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_open_repository() {
        // try to open the current git repository
        let current_dir = env::current_dir().unwrap();
        let result = CommitExtractor::new(&current_dir);

        // should succeed if we're in a git repo
        if let Ok(extractor) = result {
            assert!(!extractor.repo_name.is_empty());
        }
    }

    #[test]
    fn test_extract_commits() {
        let current_dir = env::current_dir().unwrap();
        if let Ok(extractor) = CommitExtractor::new(&current_dir) {
            // extract up to 10 commits
            let commits = extractor.extract_commits(None, Some(10)).unwrap();
            assert!(commits.len() <= 10);

            // verify commit structure
            for commit in commits {
                assert!(!commit.commit_hash.is_empty());
                assert!(!commit.metadata.repository_name.is_empty());
            }
        }
    }
}
