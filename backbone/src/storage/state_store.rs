use crate::error::Result;
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Processing status for a document
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ProcessingStatus {
    Pending,
    Processing,
    #[serde(alias = "complete")]
    Completed,
    Failed,
}

impl ProcessingStatus {
    fn as_str(&self) -> &'static str {
        match self {
            ProcessingStatus::Pending => "pending",
            ProcessingStatus::Processing => "processing",
            ProcessingStatus::Completed => "completed",
            ProcessingStatus::Failed => "failed",
        }
    }

    fn from_str(value: &str) -> Self {
        match value {
            "pending" => ProcessingStatus::Pending,
            "processing" => ProcessingStatus::Processing,
            "completed" | "complete" => ProcessingStatus::Completed,
            "failed" => ProcessingStatus::Failed,
            _ => ProcessingStatus::Failed,
        }
    }
}

/// State information for a processed document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingState {
    /// SHA256 hash of the file
    pub file_hash: String,

    /// Source file path
    pub source_path: String,

    /// File size in bytes
    #[serde(default)]
    pub file_size_bytes: u64,

    /// Last modified time
    #[serde(default)]
    pub modified_at: Option<DateTime<Utc>>,

    /// When the file was processed
    pub processed_at: DateTime<Utc>,

    /// Number of chunks created
    pub chunk_count: usize,

    /// Processing status
    pub status: ProcessingStatus,

    /// Error message if failed
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingFailure {
    pub source_path: String,
    pub file_hash: Option<String>,
    pub file_size_bytes: u64,
    pub modified_at: Option<DateTime<Utc>>,
    pub failed_at: DateTime<Utc>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ScanStatus {
    Scanning,
    Complete,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanProgress {
    pub run_id: String,
    pub files_seen: u64,
    pub bytes_seen: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanRunSummary {
    pub run_id: String,
    pub files_seen: u64,
    pub bytes_seen: u64,
    pub status: ScanStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanEntryRecord {
    pub source_path: String,
    pub file_size_bytes: u64,
    pub modified_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ScanEntriesSummary {
    pub files_seen: u64,
    pub bytes_seen: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScanRunState {
    run_id: String,
    files_seen: u64,
    bytes_seen: u64,
    status: ScanStatus,
    updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScanEntry {
    run_id: String,
    file_size_bytes: u64,
    #[serde(default)]
    modified_at: Option<DateTime<Utc>>,
    updated_at: DateTime<Utc>,
}

/// State store for tracking document processing
pub struct StateStore {
    backend: StateStoreBackend,
}

enum StateStoreBackend {
    Json(JsonStateStore),
    Sqlite(SqliteStateStore),
}

impl StateStore {
    /// Create or load a state store from the given path
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let backend = if is_sqlite_path(&path) {
            StateStoreBackend::Sqlite(SqliteStateStore::new(path)?)
        } else {
            StateStoreBackend::Json(JsonStateStore::new(path)?)
        };
        Ok(StateStore { backend })
    }

    /// Check if a source path has been processed
    pub fn is_processed(&self, source_path: &str) -> bool {
        match &self.backend {
            StateStoreBackend::Json(store) => store.is_processed(source_path),
            StateStoreBackend::Sqlite(store) => store.is_processed(source_path),
        }
    }

    /// Get processing state for a source path
    pub fn get(&self, source_path: &str) -> Option<ProcessingState> {
        match &self.backend {
            StateStoreBackend::Json(store) => store.get(source_path).cloned(),
            StateStoreBackend::Sqlite(store) => store.get(source_path).ok().flatten(),
        }
    }

    /// Check if a file should be skipped based on metadata
    pub fn should_skip(
        &self,
        source_path: &str,
        file_size_bytes: u64,
        modified_at: Option<DateTime<Utc>>,
    ) -> bool {
        match &self.backend {
            StateStoreBackend::Json(store) => {
                store.should_skip(source_path, file_size_bytes, modified_at)
            }
            StateStoreBackend::Sqlite(store) => {
                store.should_skip(source_path, file_size_bytes, modified_at)
            }
        }
    }

    /// Mark a file as currently being processed
    pub fn mark_processing(
        &mut self,
        file_hash: String,
        source_path: String,
        file_size_bytes: u64,
        modified_at: Option<DateTime<Utc>>,
    ) -> Result<()> {
        match &mut self.backend {
            StateStoreBackend::Json(store) => store.mark_processing(
                file_hash,
                source_path,
                file_size_bytes,
                modified_at,
            ),
            StateStoreBackend::Sqlite(store) => store.mark_processing(
                file_hash,
                source_path,
                file_size_bytes,
                modified_at,
            ),
        }
    }

    pub fn finalize_incomplete(&mut self, reason: &str) -> Result<usize> {
        match &mut self.backend {
            StateStoreBackend::Json(store) => store.finalize_incomplete(reason),
            StateStoreBackend::Sqlite(store) => store.finalize_incomplete(reason),
        }
    }

    /// Mark a file as completed
    pub fn mark_complete(&mut self, source_path: &str, chunk_count: usize) -> Result<()> {
        match &mut self.backend {
            StateStoreBackend::Json(store) => store.mark_complete(source_path, chunk_count),
            StateStoreBackend::Sqlite(store) => store.mark_complete(source_path, chunk_count),
        }
    }

    /// Mark a file as failed
    pub fn mark_failed(&mut self, source_path: &str, error: String) -> Result<()> {
        match &mut self.backend {
            StateStoreBackend::Json(store) => store.mark_failed(source_path, error),
            StateStoreBackend::Sqlite(store) => store.mark_failed(source_path, error),
        }
    }

    pub fn record_failure(
        &mut self,
        source_path: &str,
        file_hash: Option<String>,
        file_size_bytes: u64,
        modified_at: Option<DateTime<Utc>>,
        reason: String,
    ) -> Result<()> {
        match &mut self.backend {
            StateStoreBackend::Json(store) => store.record_failure(
                source_path,
                file_hash,
                file_size_bytes,
                modified_at,
                reason,
            ),
            StateStoreBackend::Sqlite(store) => store.record_failure(
                source_path,
                file_hash,
                file_size_bytes,
                modified_at,
                reason,
            ),
        }
    }

    pub fn clear_failure(&mut self, source_path: &str) -> Result<()> {
        match &mut self.backend {
            StateStoreBackend::Json(store) => store.clear_failure(source_path),
            StateStoreBackend::Sqlite(store) => store.clear_failure(source_path),
        }
    }

    pub fn should_skip_failure(
        &self,
        source_path: &str,
        file_size_bytes: u64,
        modified_at: Option<DateTime<Utc>>,
    ) -> bool {
        match &self.backend {
            StateStoreBackend::Json(store) => {
                store.should_skip_failure(source_path, file_size_bytes, modified_at)
            }
            StateStoreBackend::Sqlite(store) => {
                store.should_skip_failure(source_path, file_size_bytes, modified_at)
            }
        }
    }

    /// Remove a file from the state
    pub fn remove(&mut self, source_path: &str) -> Result<()> {
        match &mut self.backend {
            StateStoreBackend::Json(store) => store.remove(source_path),
            StateStoreBackend::Sqlite(store) => store.remove(source_path),
        }
    }

    /// Get all source paths in the state
    pub fn all_hashes(&self) -> Result<Vec<String>> {
        match &self.backend {
            StateStoreBackend::Json(store) => Ok(store.all_hashes()),
            StateStoreBackend::Sqlite(store) => store.all_hashes(),
        }
    }

    /// Get count of processed files
    pub fn count_completed(&self) -> Result<usize> {
        match &self.backend {
            StateStoreBackend::Json(store) => Ok(store.count_completed()),
            StateStoreBackend::Sqlite(store) => store.count_completed(),
        }
    }

    pub fn start_scan(&mut self, scope_key: &str) -> Result<ScanProgress> {
        match &mut self.backend {
            StateStoreBackend::Json(store) => store.start_scan(scope_key),
            StateStoreBackend::Sqlite(store) => store.start_scan(scope_key),
        }
    }

    pub fn get_scan_run(&self, scope_key: &str) -> Result<Option<ScanRunSummary>> {
        match &self.backend {
            StateStoreBackend::Json(store) => Ok(store.get_scan_run(scope_key)),
            StateStoreBackend::Sqlite(store) => store.get_scan_run(scope_key),
        }
    }

    pub fn list_scan_entries(
        &self,
        scope_key: &str,
        run_id: &str,
    ) -> Result<Vec<ScanEntryRecord>> {
        match &self.backend {
            StateStoreBackend::Json(store) => Ok(store.list_scan_entries(scope_key, run_id)),
            StateStoreBackend::Sqlite(store) => store.list_scan_entries(scope_key, run_id),
        }
    }

    pub fn scan_entries_summary(
        &self,
        scope_key: &str,
        run_id: &str,
    ) -> Result<ScanEntriesSummary> {
        match &self.backend {
            StateStoreBackend::Json(store) => Ok(store.scan_entries_summary(scope_key, run_id)),
            StateStoreBackend::Sqlite(store) => store.scan_entries_summary(scope_key, run_id),
        }
    }

    pub fn update_scan_run_counts(
        &mut self,
        scope_key: &str,
        run_id: &str,
        files_seen: u64,
        bytes_seen: u64,
    ) -> Result<()> {
        match &mut self.backend {
            StateStoreBackend::Json(store) => {
                store.update_scan_run_counts(scope_key, run_id, files_seen, bytes_seen)
            }
            StateStoreBackend::Sqlite(store) => {
                store.update_scan_run_counts(scope_key, run_id, files_seen, bytes_seen)
            }
        }
    }

    pub fn processed_summary_for_scan(
        &self,
        scope_key: &str,
        run_id: &str,
    ) -> Result<ScanEntriesSummary> {
        match &self.backend {
            StateStoreBackend::Json(store) => Ok(store.processed_summary_for_scan(scope_key, run_id)),
            StateStoreBackend::Sqlite(store) => store.processed_summary_for_scan(scope_key, run_id),
        }
    }

    pub fn record_scan_entry(
        &mut self,
        scope_key: &str,
        run_id: &str,
        source_path: &str,
        file_size_bytes: u64,
        modified_at: Option<DateTime<Utc>>,
    ) -> Result<ScanProgress> {
        match &mut self.backend {
            StateStoreBackend::Json(store) => {
                store.record_scan_entry(
                    scope_key,
                    run_id,
                    source_path,
                    file_size_bytes,
                    modified_at,
                )
            }
            StateStoreBackend::Sqlite(store) => {
                store.record_scan_entry(
                    scope_key,
                    run_id,
                    source_path,
                    file_size_bytes,
                    modified_at,
                )
            }
        }
    }

    pub fn finish_scan(&mut self, scope_key: &str, run_id: &str) -> Result<()> {
        match &mut self.backend {
            StateStoreBackend::Json(store) => store.finish_scan(scope_key, run_id),
            StateStoreBackend::Sqlite(store) => store.finish_scan(scope_key, run_id),
        }
    }
}

fn is_sqlite_path(path: &Path) -> bool {
    match path.extension().and_then(|value| value.to_str()) {
        Some(ext) => matches!(ext.to_lowercase().as_str(), "db" | "sqlite" | "sqlite3"),
        None => false,
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct JsonStateData {
    #[serde(default)]
    processing: HashMap<String, ProcessingState>,
    #[serde(default)]
    failures: HashMap<String, ProcessingFailure>,
    #[serde(default)]
    scan_runs: HashMap<String, ScanRunState>,
    #[serde(default)]
    scan_entries: HashMap<String, ScanEntry>,
}

struct JsonStateStore {
    path: PathBuf,
    processing: HashMap<String, ProcessingState>,
    failures: HashMap<String, ProcessingFailure>,
    scan_runs: HashMap<String, ScanRunState>,
    scan_entries: HashMap<String, ScanEntry>,
}

impl JsonStateStore {
    /// Create or load a state store from the given path
    fn new(path: PathBuf) -> Result<Self> {
        let raw_data: JsonStateData = if path.exists() {
            let content = fs::read_to_string(&path)?;
            // handle empty file
            if content.trim().is_empty() {
                JsonStateData::default()
            } else {
                let value: serde_json::Value = serde_json::from_str(&content)?;
                if value.get("processing").is_some() {
                    serde_json::from_value(value)?
                } else {
                    let legacy: HashMap<String, ProcessingState> =
                        serde_json::from_value(value)?;
                    JsonStateData {
                        processing: legacy,
                        ..JsonStateData::default()
                    }
                }
            }
        } else {
            JsonStateData::default()
        };

        let mut processing: HashMap<String, ProcessingState> = HashMap::new();
        for (_, entry) in raw_data.processing {
            let key = if entry.source_path.is_empty() {
                entry.file_hash.clone()
            } else {
                entry.source_path.clone()
            };
            let replace = match processing.get(&key) {
                Some(existing) => entry.processed_at > existing.processed_at,
                None => true,
            };
            if replace {
                processing.insert(key, entry);
            }
        }

        Ok(Self {
            path,
            processing,
            failures: raw_data.failures,
            scan_runs: raw_data.scan_runs,
            scan_entries: raw_data.scan_entries,
        })
    }

    /// Check if a source path has been processed
    fn is_processed(&self, source_path: &str) -> bool {
        self.processing
            .get(source_path)
            .map(|s| s.status == ProcessingStatus::Completed)
            .unwrap_or(false)
    }

    /// Get processing state for a source path
    fn get(&self, source_path: &str) -> Option<&ProcessingState> {
        self.processing.get(source_path)
    }

    /// Check if a file should be skipped based on metadata
    fn should_skip(
        &self,
        source_path: &str,
        file_size_bytes: u64,
        modified_at: Option<DateTime<Utc>>,
    ) -> bool {
        let state = match self.processing.get(source_path) {
            Some(state) => state,
            None => return false,
        };

        if state.status != ProcessingStatus::Completed {
            return false;
        }

        if state.file_size_bytes != file_size_bytes {
            return false;
        }

        if state.modified_at.is_some()
            && modified_at.is_some()
            && state.modified_at != modified_at
        {
            return false;
        }

        true
    }

    /// Mark a file as currently being processed
    fn mark_processing(
        &mut self,
        file_hash: String,
        source_path: String,
        file_size_bytes: u64,
        modified_at: Option<DateTime<Utc>>,
    ) -> Result<()> {
        let state = ProcessingState {
            file_hash: file_hash.clone(),
            source_path: source_path.clone(),
            file_size_bytes,
            modified_at,
            processed_at: Utc::now(),
            chunk_count: 0,
            status: ProcessingStatus::Processing,
            error: None,
        };
        self.processing.insert(source_path, state);
        self.save()
    }

    fn finalize_incomplete(&mut self, reason: &str) -> Result<usize> {
        let mut updated = 0_usize;
        let now = Utc::now();
        let reason = reason.to_string();

        for state in self.processing.values_mut() {
            if state.status != ProcessingStatus::Processing {
                continue;
            }
            state.status = ProcessingStatus::Failed;
            state.error = Some(reason.clone());
            state.processed_at = now;

            self.failures.insert(
                state.source_path.clone(),
                ProcessingFailure {
                    source_path: state.source_path.clone(),
                    file_hash: Some(state.file_hash.clone()),
                    file_size_bytes: state.file_size_bytes,
                    modified_at: state.modified_at,
                    failed_at: now,
                    reason: reason.clone(),
                },
            );
            updated += 1;
        }

        if updated > 0 {
            self.save()?;
        }
        Ok(updated)
    }

    /// Mark a file as completed
    fn mark_complete(&mut self, source_path: &str, chunk_count: usize) -> Result<()> {
        if let Some(state) = self.processing.get_mut(source_path) {
            state.status = ProcessingStatus::Completed;
            state.chunk_count = chunk_count;
            state.processed_at = Utc::now();
            state.error = None;
        }
        self.failures.remove(source_path);
        self.save()
    }

    /// Mark a file as failed
    fn mark_failed(&mut self, source_path: &str, error: String) -> Result<()> {
        if let Some(state) = self.processing.get_mut(source_path) {
            state.status = ProcessingStatus::Failed;
            state.error = Some(error);
            state.processed_at = Utc::now();
        }
        self.save()
    }

    fn record_failure(
        &mut self,
        source_path: &str,
        file_hash: Option<String>,
        file_size_bytes: u64,
        modified_at: Option<DateTime<Utc>>,
        reason: String,
    ) -> Result<()> {
        let failure = ProcessingFailure {
            source_path: source_path.to_string(),
            file_hash,
            file_size_bytes,
            modified_at,
            failed_at: Utc::now(),
            reason,
        };
        self.failures.insert(source_path.to_string(), failure);
        self.save()
    }

    fn clear_failure(&mut self, source_path: &str) -> Result<()> {
        self.failures.remove(source_path);
        self.save()
    }

    fn should_skip_failure(
        &self,
        source_path: &str,
        file_size_bytes: u64,
        modified_at: Option<DateTime<Utc>>,
    ) -> bool {
        let failure = match self.failures.get(source_path) {
            Some(failure) => failure,
            None => return false,
        };

        if failure.file_size_bytes != file_size_bytes {
            return false;
        }

        if failure.modified_at.is_some()
            && modified_at.is_some()
            && failure.modified_at != modified_at
        {
            return false;
        }

        true
    }

    /// Remove a file from the state
    fn remove(&mut self, source_path: &str) -> Result<()> {
        self.processing.remove(source_path);
        self.failures.remove(source_path);
        self.save()
    }

    /// Get all source paths in the state
    fn all_hashes(&self) -> Vec<String> {
        self.processing.keys().cloned().collect()
    }

    /// Get count of processed files
    fn count_completed(&self) -> usize {
        self.processing
            .values()
            .filter(|s| s.status == ProcessingStatus::Completed)
            .count()
    }

    fn start_scan(&mut self, scope_key: &str) -> Result<ScanProgress> {
        let now = Utc::now();
        let run_id = match self.scan_runs.get_mut(scope_key) {
            Some(existing) => {
                if existing.status == ScanStatus::Scanning {
                    existing.updated_at = now;
                    existing.run_id.clone()
                } else {
                    let run_id = new_scan_run_id(now);
                    *existing = ScanRunState {
                        run_id: run_id.clone(),
                        files_seen: 0,
                        bytes_seen: 0,
                        status: ScanStatus::Scanning,
                        updated_at: now,
                    };
                    run_id
                }
            }
            None => {
                let run_id = new_scan_run_id(now);
                self.scan_runs.insert(
                    scope_key.to_string(),
                    ScanRunState {
                        run_id: run_id.clone(),
                        files_seen: 0,
                        bytes_seen: 0,
                        status: ScanStatus::Scanning,
                        updated_at: now,
                    },
                );
                run_id
            }
        };

        let progress = self
            .scan_runs
            .get(scope_key)
            .map(|run| ScanProgress {
                run_id: run.run_id.clone(),
                files_seen: run.files_seen,
                bytes_seen: run.bytes_seen,
            })
            .unwrap_or(ScanProgress {
                run_id,
                files_seen: 0,
                bytes_seen: 0,
            });

        self.save()?;
        Ok(progress)
    }

    fn get_scan_run(&self, scope_key: &str) -> Option<ScanRunSummary> {
        self.scan_runs.get(scope_key).map(|run| ScanRunSummary {
            run_id: run.run_id.clone(),
            files_seen: run.files_seen,
            bytes_seen: run.bytes_seen,
            status: run.status.clone(),
        })
    }

    fn list_scan_entries(&self, scope_key: &str, run_id: &str) -> Vec<ScanEntryRecord> {
        let mut results = Vec::new();
        for (key, entry) in &self.scan_entries {
            let (entry_scope, source_path) = match parse_scan_entry_key(key) {
                Some(value) => value,
                None => continue,
            };
            if entry_scope != scope_key || entry.run_id != run_id {
                continue;
            }
            results.push(ScanEntryRecord {
                source_path,
                file_size_bytes: entry.file_size_bytes,
                modified_at: entry.modified_at.clone(),
            });
        }
        results
    }

    fn scan_entries_summary(&self, scope_key: &str, run_id: &str) -> ScanEntriesSummary {
        let mut files_seen = 0_u64;
        let mut bytes_seen = 0_u64;
        for (key, entry) in &self.scan_entries {
            let (entry_scope, _) = match parse_scan_entry_key(key) {
                Some(value) => value,
                None => continue,
            };
            if entry_scope != scope_key || entry.run_id != run_id {
                continue;
            }
            files_seen += 1;
            bytes_seen = bytes_seen.saturating_add(entry.file_size_bytes);
        }
        ScanEntriesSummary {
            files_seen,
            bytes_seen,
        }
    }

    fn update_scan_run_counts(
        &mut self,
        scope_key: &str,
        run_id: &str,
        files_seen: u64,
        bytes_seen: u64,
    ) -> Result<()> {
        if let Some(run) = self.scan_runs.get_mut(scope_key) {
            if run.run_id == run_id {
                run.files_seen = files_seen;
                run.bytes_seen = bytes_seen;
                run.updated_at = Utc::now();
                self.save()?;
            }
        }
        Ok(())
    }

    fn processed_summary_for_scan(&self, scope_key: &str, run_id: &str) -> ScanEntriesSummary {
        let mut files_seen = 0_u64;
        let mut bytes_seen = 0_u64;
        for (key, entry) in &self.scan_entries {
            let (entry_scope, source_path) = match parse_scan_entry_key(key) {
                Some(value) => value,
                None => continue,
            };
            if entry_scope != scope_key || entry.run_id != run_id {
                continue;
            }

            let mut handled = false;
            if let Some(state) = self.processing.get(&source_path) {
                if state.status == ProcessingStatus::Completed
                    && state.file_size_bytes == entry.file_size_bytes
                    && !(state.modified_at.is_some()
                        && entry.modified_at.is_some()
                        && state.modified_at != entry.modified_at)
                {
                    handled = true;
                }
            }

            if !handled {
                if let Some(failure) = self.failures.get(&source_path) {
                    if failure.file_size_bytes == entry.file_size_bytes
                        && !(failure.modified_at.is_some()
                            && entry.modified_at.is_some()
                            && failure.modified_at != entry.modified_at)
                    {
                        handled = true;
                    }
                }
            }

            if !handled {
                continue;
            }

            files_seen += 1;
            bytes_seen = bytes_seen.saturating_add(entry.file_size_bytes);
        }

        ScanEntriesSummary {
            files_seen,
            bytes_seen,
        }
    }

    fn record_scan_entry(
        &mut self,
        scope_key: &str,
        run_id: &str,
        source_path: &str,
        file_size_bytes: u64,
        modified_at: Option<DateTime<Utc>>,
    ) -> Result<ScanProgress> {
        let now = Utc::now();
        let entry_key = scan_entry_key(scope_key, source_path);
        let mut files_delta = 0_u64;
        let mut bytes_delta = 0_i64;

        match self.scan_entries.get_mut(&entry_key) {
            Some(entry) => {
                if entry.run_id == run_id {
                    let modified_changed = entry.modified_at != modified_at;
                    if entry.file_size_bytes != file_size_bytes {
                        bytes_delta =
                            file_size_bytes as i64 - entry.file_size_bytes as i64;
                        entry.file_size_bytes = file_size_bytes;
                        entry.modified_at = modified_at;
                        entry.updated_at = now;
                    } else if modified_changed {
                        entry.modified_at = modified_at;
                        entry.updated_at = now;
                    }
                } else {
                    files_delta = 1;
                    bytes_delta = file_size_bytes as i64;
                    entry.run_id = run_id.to_string();
                    entry.file_size_bytes = file_size_bytes;
                    entry.modified_at = modified_at;
                    entry.updated_at = now;
                }
            }
            None => {
                files_delta = 1;
                bytes_delta = file_size_bytes as i64;
                self.scan_entries.insert(
                    entry_key,
                    ScanEntry {
                        run_id: run_id.to_string(),
                        file_size_bytes,
                        modified_at,
                        updated_at: now,
                    },
                );
            }
        }

        if let Some(run) = self.scan_runs.get_mut(scope_key) {
            run.files_seen = run.files_seen.saturating_add(files_delta);
            if bytes_delta >= 0 {
                run.bytes_seen = run.bytes_seen.saturating_add(bytes_delta as u64);
            } else {
                let delta = (-bytes_delta) as u64;
                run.bytes_seen = run
                    .bytes_seen
                    .saturating_sub(delta);
            }
            run.updated_at = now;
            run.status = ScanStatus::Scanning;
        }

        self.save()?;
        Ok(self
            .scan_runs
            .get(scope_key)
            .map(|run| ScanProgress {
                run_id: run.run_id.clone(),
                files_seen: run.files_seen,
                bytes_seen: run.bytes_seen,
            })
            .unwrap_or(ScanProgress {
                run_id: run_id.to_string(),
                files_seen: 0,
                bytes_seen: 0,
            }))
    }

    fn finish_scan(&mut self, scope_key: &str, run_id: &str) -> Result<()> {
        if let Some(run) = self.scan_runs.get_mut(scope_key) {
            if run.run_id == run_id {
                run.status = ScanStatus::Complete;
                run.updated_at = Utc::now();
                let prefix = scan_entry_prefix(scope_key);
                self.scan_entries
                    .retain(|key, entry| !key.starts_with(&prefix) || entry.run_id == run_id);
                self.save()?;
            }
        }
        Ok(())
    }

    /// Save state to disk
    fn save(&self) -> Result<()> {
        // ensure parent directory exists
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }

        let data = JsonStateData {
            processing: self.processing.clone(),
            failures: self.failures.clone(),
            scan_runs: self.scan_runs.clone(),
            scan_entries: self.scan_entries.clone(),
        };
        let json = serde_json::to_string_pretty(&data)?;
        let tmp_path = self.path.with_extension("tmp");
        fs::write(&tmp_path, json)?;
        fs::rename(&tmp_path, &self.path)?;
        Ok(())
    }
}

struct SqliteStateStore {
    _path: PathBuf,
    conn: Connection,
}

impl SqliteStateStore {
    fn new(path: PathBuf) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(&path)?;
        conn.busy_timeout(Duration::from_secs(5))?;
        conn.execute_batch(
            r#"
            pragma journal_mode=WAL;
            create table if not exists processing_state (
              source_path text primary key,
              file_hash text not null,
              file_size_bytes integer not null,
              modified_at text,
              processed_at text not null,
              chunk_count integer not null,
              status text not null,
              error text
            );
            create index if not exists processing_state_status_idx
              on processing_state(status);
            create table if not exists scan_runs (
              scope_key text primary key,
              run_id text not null,
              files_seen integer not null,
              bytes_seen integer not null,
              status text not null,
              updated_at text not null
            );
            create table if not exists scan_entries (
              scope_key text not null,
              source_path text not null,
              run_id text not null,
              file_size_bytes integer not null,
              modified_at text,
              updated_at text not null,
              primary key (scope_key, source_path)
            );
            create table if not exists processing_failures (
              source_path text primary key,
              file_hash text,
              file_size_bytes integer not null,
              modified_at text,
              failed_at text not null,
              reason text not null
            );
            create index if not exists processing_failures_failed_at_idx
              on processing_failures(failed_at);
            create index if not exists scan_entries_scope_idx
              on scan_entries(scope_key);
            create index if not exists scan_entries_run_idx
              on scan_entries(scope_key, run_id);
            "#,
        )?;
        if conn
            .prepare("select modified_at from scan_entries limit 1")
            .is_err()
        {
            let _ = conn.execute("alter table scan_entries add column modified_at text", []);
        }
        if conn
            .prepare("select reason from processing_failures limit 1")
            .is_err()
        {
            let _ = conn.execute(
                "create table if not exists processing_failures (\
                 source_path text primary key, \
                 file_hash text, \
                 file_size_bytes integer not null, \
                 modified_at text, \
                 failed_at text not null, \
                 reason text not null\
                 )",
                [],
            );
        }

        Ok(Self { _path: path, conn })
    }

    fn is_processed(&self, source_path: &str) -> bool {
        let status: Option<String> = match self
            .conn
            .query_row(
                "select status from processing_state where source_path = ?1",
                params![source_path],
                |row| row.get(0),
            )
            .optional()
        {
            Ok(status) => status,
            Err(_) => None,
        };

        status
            .map(|value| ProcessingStatus::from_str(&value) == ProcessingStatus::Completed)
            .unwrap_or(false)
    }

    fn get(&self, source_path: &str) -> Result<Option<ProcessingState>> {
        let row = self
            .conn
            .query_row(
                "select file_hash, source_path, file_size_bytes, modified_at, \
                        processed_at, chunk_count, status, error\
                 from processing_state where source_path = ?1",
                params![source_path],
                |row| {
                    let file_hash: String = row.get(0)?;
                    let source_path: String = row.get(1)?;
                    let file_size_bytes: u64 = row.get(2)?;
                    let modified_at: Option<String> = row.get(3)?;
                    let processed_at: String = row.get(4)?;
                    let chunk_count: usize = row.get(5)?;
                    let status: String = row.get(6)?;
                    let error: Option<String> = row.get(7)?;

                    let modified_at = parse_datetime(modified_at);
                    let processed_at = parse_datetime(Some(processed_at))
                        .unwrap_or_else(Utc::now);

                    Ok(ProcessingState {
                        file_hash,
                        source_path,
                        file_size_bytes,
                        modified_at,
                        processed_at,
                        chunk_count,
                        status: ProcessingStatus::from_str(&status),
                        error,
                    })
                },
            )
            .optional()?;

        Ok(row)
    }

    fn should_skip(
        &self,
        source_path: &str,
        file_size_bytes: u64,
        modified_at: Option<DateTime<Utc>>,
    ) -> bool {
        let row: Option<(u64, Option<String>, String)> = match self
            .conn
            .query_row(
                "select file_size_bytes, modified_at, status \
                 from processing_state where source_path = ?1",
                params![source_path],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()
        {
            Ok(value) => value,
            Err(_) => None,
        };

        let (stored_size, stored_modified_at, status) = match row {
            Some(value) => value,
            None => return false,
        };

        if ProcessingStatus::from_str(&status) != ProcessingStatus::Completed {
            return false;
        }

        if stored_size != file_size_bytes {
            return false;
        }

        let stored_modified_at = parse_datetime(stored_modified_at);
        if stored_modified_at.is_some()
            && modified_at.is_some()
            && stored_modified_at != modified_at
        {
            return false;
        }

        true
    }

    fn mark_processing(
        &mut self,
        file_hash: String,
        source_path: String,
        file_size_bytes: u64,
        modified_at: Option<DateTime<Utc>>,
    ) -> Result<()> {
        let modified_at = modified_at.map(|value| value.to_rfc3339());
        let processed_at = Utc::now().to_rfc3339();

        self.conn.execute(
            r#"
            insert into processing_state (
                source_path, file_hash, file_size_bytes, modified_at, processed_at,
                chunk_count, status, error
            ) values (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            on conflict(source_path) do update set
                file_hash = excluded.file_hash,
                file_size_bytes = excluded.file_size_bytes,
                modified_at = excluded.modified_at,
                processed_at = excluded.processed_at,
                chunk_count = excluded.chunk_count,
                status = excluded.status,
                error = excluded.error
            "#,
            params![
                source_path,
                file_hash,
                file_size_bytes,
                modified_at,
                processed_at,
                0_u64,
                ProcessingStatus::Processing.as_str(),
                Option::<String>::None,
            ],
        )?;
        Ok(())
    }

    fn finalize_incomplete(&mut self, reason: &str) -> Result<usize> {
        let now = Utc::now().to_rfc3339();
        let reason = reason.to_string();

        let updated = self.conn.execute(
            "update processing_state set status = ?1, error = ?2, processed_at = ?3 \
             where status = ?4",
            params![
                ProcessingStatus::Failed.as_str(),
                reason,
                now,
                ProcessingStatus::Processing.as_str(),
            ],
        )? as usize;

        if updated > 0 {
            let _ = self.conn.execute(
                r#"
                insert into processing_failures (
                    source_path, file_hash, file_size_bytes, modified_at, failed_at, reason
                )
                select source_path, file_hash, file_size_bytes, modified_at, ?1, ?2
                from processing_state
                where status = ?3 and error = ?2 and processed_at = ?1
                on conflict(source_path) do update set
                    file_hash = excluded.file_hash,
                    file_size_bytes = excluded.file_size_bytes,
                    modified_at = excluded.modified_at,
                    failed_at = excluded.failed_at,
                    reason = excluded.reason
                "#,
                params![now, reason, ProcessingStatus::Failed.as_str()],
            );
        }

        Ok(updated)
    }

    fn mark_complete(&mut self, source_path: &str, chunk_count: usize) -> Result<()> {
        let processed_at = Utc::now().to_rfc3339();
        self.conn.execute(
            "update processing_state set\
                processed_at = ?1,\
                chunk_count = ?2,\
                status = ?3,\
                error = null\
             where source_path = ?4",
            params![
                processed_at,
                chunk_count as u64,
                ProcessingStatus::Completed.as_str(),
                source_path,
            ],
        )?;
        let _ = self.conn.execute(
            "delete from processing_failures where source_path = ?1",
            params![source_path],
        );
        Ok(())
    }

    fn mark_failed(&mut self, source_path: &str, error: String) -> Result<()> {
        let processed_at = Utc::now().to_rfc3339();
        self.conn.execute(
            "update processing_state set\
                processed_at = ?1,\
                status = ?2,\
                error = ?3\
             where source_path = ?4",
            params![
                processed_at,
                ProcessingStatus::Failed.as_str(),
                error,
                source_path,
            ],
        )?;
        Ok(())
    }

    fn record_failure(
        &mut self,
        source_path: &str,
        file_hash: Option<String>,
        file_size_bytes: u64,
        modified_at: Option<DateTime<Utc>>,
        reason: String,
    ) -> Result<()> {
        let failed_at = Utc::now().to_rfc3339();
        let modified_at = modified_at.map(|value| value.to_rfc3339());
        self.conn.execute(
            r#"
            insert into processing_failures (
                source_path, file_hash, file_size_bytes, modified_at, failed_at, reason
            ) values (?1, ?2, ?3, ?4, ?5, ?6)
            on conflict(source_path) do update set
                file_hash = excluded.file_hash,
                file_size_bytes = excluded.file_size_bytes,
                modified_at = excluded.modified_at,
                failed_at = excluded.failed_at,
                reason = excluded.reason
            "#,
            params![
                source_path,
                file_hash,
                file_size_bytes,
                modified_at,
                failed_at,
                reason,
            ],
        )?;
        Ok(())
    }

    fn clear_failure(&mut self, source_path: &str) -> Result<()> {
        self.conn.execute(
            "delete from processing_failures where source_path = ?1",
            params![source_path],
        )?;
        Ok(())
    }

    fn should_skip_failure(
        &self,
        source_path: &str,
        file_size_bytes: u64,
        modified_at: Option<DateTime<Utc>>,
    ) -> bool {
        let row: Option<(u64, Option<String>)> = match self
            .conn
            .query_row(
                "select file_size_bytes, modified_at \
                 from processing_failures where source_path = ?1",
                params![source_path],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
        {
            Ok(value) => value,
            Err(_) => None,
        };

        let (stored_size, stored_modified_at) = match row {
            Some(value) => value,
            None => return false,
        };

        if stored_size != file_size_bytes {
            return false;
        }

        let stored_modified_at = parse_datetime(stored_modified_at);
        if stored_modified_at.is_some()
            && modified_at.is_some()
            && stored_modified_at != modified_at
        {
            return false;
        }

        true
    }

    fn remove(&mut self, source_path: &str) -> Result<()> {
        self.conn.execute(
            "delete from processing_state where source_path = ?1",
            params![source_path],
        )?;
        let _ = self.conn.execute(
            "delete from processing_failures where source_path = ?1",
            params![source_path],
        );
        Ok(())
    }

    fn all_hashes(&self) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("select source_path from processing_state")?;
        let rows = stmt.query_map([], |row| row.get(0))?;
        let mut results = Vec::new();
        for row in rows {
            if let Ok(value) = row {
                results.push(value);
            }
        }
        Ok(results)
    }

    fn count_completed(&self) -> Result<usize> {
        let count: u64 = self.conn.query_row(
            "select count(*) from processing_state where status = ?1",
            params![ProcessingStatus::Completed.as_str()],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }

    fn get_scan_run(&self, scope_key: &str) -> Result<Option<ScanRunSummary>> {
        let row: Option<(String, u64, u64, String)> = self
            .conn
            .query_row(
                "select run_id, files_seen, bytes_seen, status \
                 from scan_runs where scope_key = ?1",
                params![scope_key],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .optional()?;

        Ok(row.map(|(run_id, files_seen, bytes_seen, status)| ScanRunSummary {
            run_id,
            files_seen,
            bytes_seen,
            status: if status == "complete" {
                ScanStatus::Complete
            } else {
                ScanStatus::Scanning
            },
        }))
    }

    fn list_scan_entries(
        &self,
        scope_key: &str,
        run_id: &str,
    ) -> Result<Vec<ScanEntryRecord>> {
        let mut stmt = self.conn.prepare(
            "select source_path, file_size_bytes, modified_at \
             from scan_entries where scope_key = ?1 and run_id = ?2",
        )?;
        let rows = stmt.query_map(params![scope_key, run_id], |row| {
            let source_path: String = row.get(0)?;
            let file_size_bytes: u64 = row.get(1)?;
            let modified_at: Option<String> = row.get(2)?;
            Ok(ScanEntryRecord {
                source_path,
                file_size_bytes,
                modified_at: parse_datetime(modified_at),
            })
        })?;

        let mut results = Vec::new();
        for row in rows {
            if let Ok(value) = row {
                results.push(value);
            }
        }
        Ok(results)
    }

    fn scan_entries_summary(
        &self,
        scope_key: &str,
        run_id: &str,
    ) -> Result<ScanEntriesSummary> {
        let (files_seen, bytes_seen): (u64, u64) = self.conn.query_row(
            "select count(*), coalesce(sum(file_size_bytes), 0) \
             from scan_entries where scope_key = ?1 and run_id = ?2",
            params![scope_key, run_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        Ok(ScanEntriesSummary {
            files_seen,
            bytes_seen,
        })
    }

    fn update_scan_run_counts(
        &mut self,
        scope_key: &str,
        run_id: &str,
        files_seen: u64,
        bytes_seen: u64,
    ) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "update scan_runs set files_seen = ?1, bytes_seen = ?2, updated_at = ?3 \
             where scope_key = ?4 and run_id = ?5",
            params![files_seen, bytes_seen, now, scope_key, run_id],
        )?;
        Ok(())
    }

    fn processed_summary_for_scan(
        &self,
        scope_key: &str,
        run_id: &str,
    ) -> Result<ScanEntriesSummary> {
        let legacy_completed = "complete";
        let (files_seen, bytes_seen): (u64, u64) = self.conn.query_row(
            "select count(*), coalesce(sum(file_size_bytes), 0) from (\
             select se.source_path, se.file_size_bytes \
             from scan_entries se \
             join processing_state ps on ps.source_path = se.source_path \
             where se.scope_key = ?1 and se.run_id = ?2 \
               and ps.status in (?3, ?4) \
               and ps.file_size_bytes = se.file_size_bytes \
               and (ps.modified_at = se.modified_at \
                    or ps.modified_at is null \
                    or se.modified_at is null)\
             union \
             select se.source_path, se.file_size_bytes \
             from scan_entries se \
             join processing_failures pf on pf.source_path = se.source_path \
             where se.scope_key = ?1 and se.run_id = ?2 \
               and pf.file_size_bytes = se.file_size_bytes \
               and (pf.modified_at = se.modified_at \
                    or pf.modified_at is null \
                    or se.modified_at is null)\
             )",
            params![
                scope_key,
                run_id,
                ProcessingStatus::Completed.as_str(),
                legacy_completed
            ],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        Ok(ScanEntriesSummary {
            files_seen,
            bytes_seen,
        })
    }

    fn start_scan(&mut self, scope_key: &str) -> Result<ScanProgress> {
        let now = Utc::now();
        let existing: Option<(String, u64, u64, String)> = self
            .conn
            .query_row(
                "select run_id, files_seen, bytes_seen, status \
                 from scan_runs where scope_key = ?1",
                params![scope_key],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .optional()?;

        if let Some((run_id, files_seen, bytes_seen, status)) = existing {
            if status == "scanning" {
                self.conn.execute(
                    "update scan_runs set updated_at = ?1 where scope_key = ?2",
                    params![now.to_rfc3339(), scope_key],
                )?;
                return Ok(ScanProgress {
                    run_id,
                    files_seen,
                    bytes_seen,
                });
            }
        }

        let run_id = new_scan_run_id(now);
        self.conn.execute(
            r#"
            insert into scan_runs (scope_key, run_id, files_seen, bytes_seen, status, updated_at)
            values (?1, ?2, ?3, ?4, ?5, ?6)
            on conflict(scope_key) do update set
              run_id = excluded.run_id,
              files_seen = excluded.files_seen,
              bytes_seen = excluded.bytes_seen,
              status = excluded.status,
              updated_at = excluded.updated_at
            "#,
            params![
                scope_key,
                run_id,
                0_u64,
                0_u64,
                "scanning",
                now.to_rfc3339(),
            ],
        )?;

        Ok(ScanProgress {
            run_id,
            files_seen: 0,
            bytes_seen: 0,
        })
    }

    fn record_scan_entry(
        &mut self,
        scope_key: &str,
        run_id: &str,
        source_path: &str,
        file_size_bytes: u64,
        modified_at: Option<DateTime<Utc>>,
    ) -> Result<ScanProgress> {
        let now = Utc::now();
        let existing: Option<(String, u64, Option<String>)> = self
            .conn
            .query_row(
                "select run_id, file_size_bytes, modified_at from scan_entries \
                 where scope_key = ?1 and source_path = ?2",
                params![scope_key, source_path],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        let modified_at = modified_at.map(|value| value.to_rfc3339());

        let mut files_delta = 0_u64;
        let mut bytes_delta = 0_i64;

        match existing {
            Some((existing_run_id, existing_size, existing_modified_at)) => {
                if existing_run_id == run_id {
                    if existing_size != file_size_bytes
                        || existing_modified_at != modified_at
                    {
                        bytes_delta =
                            file_size_bytes as i64 - existing_size as i64;
                        self.conn.execute(
                            "update scan_entries set file_size_bytes = ?1, modified_at = ?2, \
                             updated_at = ?3 \
                             where scope_key = ?4 and source_path = ?5",
                            params![
                                file_size_bytes,
                                modified_at,
                                now.to_rfc3339(),
                                scope_key,
                                source_path,
                            ],
                        )?;
                    }
                } else {
                    files_delta = 1;
                    bytes_delta = file_size_bytes as i64;
                    self.conn.execute(
                        "update scan_entries set run_id = ?1, file_size_bytes = ?2, \
                         modified_at = ?3, updated_at = ?4 \
                         where scope_key = ?5 and source_path = ?6",
                        params![
                            run_id,
                            file_size_bytes,
                            modified_at,
                            now.to_rfc3339(),
                            scope_key,
                            source_path,
                        ],
                    )?;
                }
            }
            None => {
                files_delta = 1;
                bytes_delta = file_size_bytes as i64;
                self.conn.execute(
                    "insert into scan_entries (scope_key, source_path, run_id, file_size_bytes, modified_at, updated_at) \
                     values (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![
                        scope_key,
                        source_path,
                        run_id,
                        file_size_bytes,
                        modified_at,
                        now.to_rfc3339(),
                    ],
                )?;
            }
        }

        let (mut files_seen, mut bytes_seen): (u64, u64) = self.conn.query_row(
            "select files_seen, bytes_seen from scan_runs where scope_key = ?1",
            params![scope_key],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;

        files_seen = files_seen.saturating_add(files_delta);
        if bytes_delta >= 0 {
            bytes_seen = bytes_seen.saturating_add(bytes_delta as u64);
        } else {
            let delta = (-bytes_delta) as u64;
            bytes_seen = bytes_seen.saturating_sub(delta);
        }

        self.conn.execute(
            "update scan_runs set files_seen = ?1, bytes_seen = ?2, \
             status = ?3, updated_at = ?4 where scope_key = ?5",
            params![
                files_seen,
                bytes_seen,
                "scanning",
                now.to_rfc3339(),
                scope_key,
            ],
        )?;

        Ok(ScanProgress {
            run_id: run_id.to_string(),
            files_seen,
            bytes_seen,
        })
    }

    fn finish_scan(&mut self, scope_key: &str, run_id: &str) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "update scan_runs set status = ?1, updated_at = ?2 \
             where scope_key = ?3 and run_id = ?4",
            params!["complete", now, scope_key, run_id],
        )?;
        self.conn.execute(
            "delete from scan_entries where scope_key = ?1 and run_id != ?2",
            params![scope_key, run_id],
        )?;
        Ok(())
    }
}

fn parse_datetime(value: Option<String>) -> Option<DateTime<Utc>> {
    value.and_then(|raw| {
        DateTime::parse_from_rfc3339(&raw)
            .ok()
            .map(|parsed| parsed.with_timezone(&Utc))
    })
}

fn new_scan_run_id(now: DateTime<Utc>) -> String {
    format!("scan-{}-{}", now.timestamp(), now.timestamp_subsec_nanos())
}

fn scan_entry_key(scope_key: &str, source_path: &str) -> String {
    format!("{}\t{}", scope_key, source_path)
}

fn scan_entry_prefix(scope_key: &str) -> String {
    format!("{}\t", scope_key)
}

fn parse_scan_entry_key(key: &str) -> Option<(String, String)> {
    let (scope, source) = key.split_once('\t')?;
    Some((scope.to_string(), source.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::{NamedTempFile, TempDir};

    fn json_store() -> StateStore {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().with_extension("json");
        StateStore::new(path).unwrap()
    }

    fn sqlite_store() -> (TempDir, StateStore) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("state.db");
        let store = StateStore::new(path).unwrap();
        (dir, store)
    }

    fn exercise_store(mut store: StateStore) -> Result<()> {
        let modified_at = Utc::now();
        store.mark_processing(
            "hash123".to_string(),
            "/path/to/file".to_string(),
            123,
            Some(modified_at),
        )?;
        store.mark_complete("/path/to/file", 10)?;

        assert!(store.is_processed("/path/to/file"));
        assert!(store.should_skip("/path/to/file", 123, Some(modified_at)));
        assert_eq!(store.count_completed()?, 1);

        store.mark_failed("/path/to/file", "test error".to_string())?;
        assert!(!store.is_processed("/path/to/file"));

        store.remove("/path/to/file")?;
        assert!(!store.is_processed("/path/to/file"));

        Ok(())
    }

    fn exercise_scan(mut store: StateStore) -> Result<()> {
        let scope_key = "scope:test";
        let progress = store.start_scan(scope_key)?;
        assert_eq!(progress.files_seen, 0);
        assert_eq!(progress.bytes_seen, 0);

        let progress = store.record_scan_entry(
            scope_key,
            &progress.run_id,
            "/path/a",
            10,
            None,
        )?;
        assert_eq!(progress.files_seen, 1);
        assert_eq!(progress.bytes_seen, 10);

        let progress = store.record_scan_entry(
            scope_key,
            &progress.run_id,
            "/path/b",
            5,
            None,
        )?;
        assert_eq!(progress.files_seen, 2);
        assert_eq!(progress.bytes_seen, 15);

        let resumed = store.start_scan(scope_key)?;
        assert_eq!(resumed.files_seen, 2);
        assert_eq!(resumed.bytes_seen, 15);

        store.finish_scan(scope_key, &resumed.run_id)?;
        let restarted = store.start_scan(scope_key)?;
        assert_eq!(restarted.files_seen, 0);
        assert_eq!(restarted.bytes_seen, 0);

        Ok(())
    }

    fn exercise_incomplete(mut store: StateStore) -> Result<()> {
        let modified_at = Utc::now();
        store.mark_processing(
            "hash123".to_string(),
            "/path/incomplete".to_string(),
            42,
            Some(modified_at),
        )?;

        let updated = store.finalize_incomplete("interrupted")?;
        assert_eq!(updated, 1);
        assert!(store.should_skip_failure("/path/incomplete", 42, Some(modified_at)));
        assert!(!store.should_skip("/path/incomplete", 42, Some(modified_at)));
        Ok(())
    }

    #[test]
    fn test_json_state_store() -> Result<()> {
        let store = json_store();
        exercise_store(store)?;
        exercise_scan(json_store())?;
        exercise_incomplete(json_store())
    }

    #[test]
    fn test_sqlite_state_store() -> Result<()> {
        let (_dir, store) = sqlite_store();
        exercise_store(store)?;
        let (_dir, store) = sqlite_store();
        exercise_scan(store)?;
        let (_dir, store) = sqlite_store();
        exercise_incomplete(store)
    }
}
