use anyhow::{anyhow, Result};
use clap::{Args, Parser, Subcommand};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

#[derive(Parser)]
#[command(name = "bbt")]
#[command(about = "big brain time - document ingestion and RAG", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate database schema from JSON payload
    Gen {
        #[command(subcommand)]
        subcommand: GenCommands,
    },
    /// Sync documents from a directory
    Sync(SyncCommand),
    /// Query indexed documents
    Query(QueryCommand),
}

struct QueryResultOutput {
    id: String,
    score: f32,
    text: String,
    citation: String,
    source: String,
    metadata: std::collections::HashMap<String, String>,
}

struct EmbeddingTask {
    file_path: PathBuf,
    source_path: String,
    display_name: String,
    file_size_bytes: u64,
    modified_at: Option<chrono::DateTime<chrono::Utc>>,
    file_hash: String,
    chunks: Vec<String>,
    /// total bytes across all chunks for adaptive batch sizing
    total_chunk_bytes: usize,
}

struct EmbeddingResult {
    task: EmbeddingTask,
    embeddings: std::result::Result<Vec<Vec<f32>>, String>,
}

#[derive(Args)]
pub struct SyncCommand {
    /// Paths to directories to sync
    #[clap(value_parser, required = true)]
    paths: Vec<PathBuf>,

    /// File extensions to include (use 'code' for source code preset)
    #[clap(long, value_delimiter = ',', default_value = "code")]
    ext: Vec<String>,

    /// Only scan git repositories
    #[clap(long, default_value = "true")]
    git_only: bool,

    /// Honor .gitignore files
    #[clap(long, default_value = "true")]
    honor_gitignore: bool,

    /// Force reprocessing even if unchanged
    #[clap(long, env = "SYNC_FORCE", default_value = "false")]
    force: bool,

    /// Reset the processing state before syncing
    #[clap(long, env = "SYNC_RESET_STATE", default_value = "false")]
    reset_state: bool,

    /// Index git commits in addition to documents
    #[clap(long, default_value = "false")]
    commits: bool,

    /// Maximum number of commits to index per repository
    #[clap(long, default_value = "1000")]
    max_commits: usize,

    /// Only index commits since this date (ISO 8601 format: YYYY-MM-DD)
    #[clap(long)]
    commits_since: Option<String>,

    /// Files larger than this (in KiB) get exclusive GPU access
    #[clap(long, default_value = "20")]
    large_file_threshold_kib: usize,

    /// Max bytes in-flight across all sessions (in KiB, max 150)
    #[clap(long, default_value = "128")]
    max_pool_bytes_kib: usize,
}

#[derive(Args)]
pub struct QueryCommand {
    /// Search query
    query: String,

    /// Number of results
    #[clap(short = 'k', long, default_value = "5")]
    top_k: usize,

    /// Retrieval mode
    #[clap(short, long, value_enum, default_value = "hybrid")]
    mode: QueryMode,

    /// Output format
    #[clap(short, long, value_enum, default_value = "text")]
    format: OutputFormat,

    /// Show scores and debug info
    #[clap(short, long)]
    verbose: bool,

    /// Output as JSON (shorthand for --format json)
    #[clap(long)]
    json: bool,

    /// Filter by source path prefix
    #[clap(long)]
    source: Option<String>,

    // --- Commit options ---
    /// Query commits instead of documents
    #[clap(long)]
    commits: bool,

    /// Filter by commit type (feat, fix, refactor, docs, test, chore)
    #[clap(long = "type")]
    commit_type: Option<String>,

    /// Filter by author email
    #[clap(long)]
    author: Option<String>,

    /// Filter since date (YYYY-MM-DD)
    #[clap(long)]
    since: Option<String>,

    /// Filter until date (YYYY-MM-DD)
    #[clap(long)]
    until: Option<String>,

    /// Sort by date (newest first) instead of score
    #[clap(long)]
    date_sort: bool,
}

#[derive(clap::ValueEnum, Debug, Clone, Copy)]
enum QueryMode {
    Vector,
    Bm25,
    Hybrid,
}

#[derive(clap::ValueEnum, Debug, Clone, Copy)]
enum CitationMode {
    Full,
    Basename,
    Hash,
    Redact,
}

#[derive(clap::ValueEnum, Debug, Clone, Copy)]
enum OutputFormat {
    Text,
    Json,
}

impl From<QueryMode> for backbone::config::RetrievalMode {
    fn from(value: QueryMode) -> Self {
        match value {
            QueryMode::Vector => backbone::config::RetrievalMode::Vector,
            QueryMode::Bm25 => backbone::config::RetrievalMode::Bm25,
            QueryMode::Hybrid => backbone::config::RetrievalMode::Hybrid,
        }
    }
}

#[derive(Subcommand)]
enum GenCommands {
    /// Generate SQL schema from JSON payload
    Schema {
        /// Input JSON file path
        #[arg(short, long)]
        input: PathBuf,

        /// Output SQL file path
        #[arg(short, long)]
        output: PathBuf,

        /// JSON Schema output path (default: <input>.schema.json)
        #[arg(long)]
        json_schema_output: Option<PathBuf>,

        /// Model repository ID
        #[arg(long, env = "GEN_SCHEMA_REPO_ID")]
        repo_id: Option<String>,

        /// Model filename
        #[arg(long, env = "GEN_SCHEMA_FILENAME")]
        filename: Option<String>,

        /// Model cache directory
        #[arg(long, env = "MODEL_CACHE_DIR")]
        cache_dir: Option<PathBuf>,
    },
    /// Generate JSON Schema from JSON payload
    JsonSchema {
        /// Input JSON file path
        #[arg(short, long)]
        input: PathBuf,

        /// Output JSON Schema file path
        #[arg(short, long)]
        output: PathBuf,

        /// Model repository ID
        #[arg(long, env = "GEN_SCHEMA_REPO_ID")]
        repo_id: Option<String>,

        /// Model filename
        #[arg(long, env = "GEN_SCHEMA_FILENAME")]
        filename: Option<String>,

        /// Model cache directory
        #[arg(long, env = "MODEL_CACHE_DIR")]
        cache_dir: Option<PathBuf>,
    },
}

pub fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Commands::Gen { subcommand } => match subcommand {
            GenCommands::Schema {
                input,
                output,
                json_schema_output,
                repo_id,
                filename,
                cache_dir,
            } => {
                let rt = tokio::runtime::Runtime::new()?;
                rt.block_on(generate_schema(
                    input,
                    output,
                    json_schema_output,
                    repo_id,
                    filename,
                    cache_dir,
                ))
            }
            GenCommands::JsonSchema {
                input,
                output,
                repo_id,
                filename,
                cache_dir,
            } => {
                let rt = tokio::runtime::Runtime::new()?;
                rt.block_on(generate_json_schema(
                    input, output, repo_id, filename, cache_dir,
                ))
            }
        },
        Commands::Sync(args) => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(sync_documents(args))
        }
        Commands::Query(args) => {
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(query_documents(args))
        }
    }
}

/// expand extension presets into actual file extensions
fn expand_extensions(ext_list: &[String]) -> Vec<String> {
    let mut extensions: Vec<String> = Vec::new();

    for ext in ext_list {
        match ext.to_lowercase().as_str() {
            "code" => {
                // source code preset: comprehensive language support
                let code_exts = [
                    // javascript/typescript
                    "js", "jsx", "ts", "tsx", "mjs", "cjs",
                    // rust
                    "rs",
                    // golang
                    "go", "mod", "sum",
                    // python
                    "py", "pyi", "pyx",
                    // c/c++/clang
                    "c", "cc", "cpp", "cxx", "h", "hh", "hpp", "hxx", "inl",
                    // c# / f# / vb
                    "cs", "fs", "fsx", "fsi", "vb",
                    // assembly
                    "asm", "s", "S",
                    // bash/shell
                    "sh", "bash", "zsh", "fish",
                    // nix
                    "nix",
                    // terraform/hcl
                    "tf", "tfvars", "hcl",
                    // ios development
                    "swift", "m", "mm", "xib", "storyboard", "plist", "xcconfig",
                    // android development
                    "kt", "kts", "java", "xml", "gradle", "pro",
                    // jvm languages
                    "scala", "sc", "clj", "cljs", "cljc", "edn",
                    // functional languages
                    "hs", "lhs", "ml", "mli", "re", "rei", "ex", "exs", "erl", "hrl",
                    // web languages
                    "php", "rb", "rake", "gemspec",
                    // scripting
                    "lua", "pl", "pm", "perl",
                    // scientific/data
                    "jl", "r", "R", "Rmd",
                    // systems languages
                    "zig", "v", "dart",
                    // blockchain
                    "sol", "move",
                    // build systems
                    "cmake", "mk", "ninja", "bzl", "bazel",
                    // visual studio / msbuild
                    "vcxproj", "sln", "props", "targets", "csproj", "vbproj", "fsproj",
                    // linker and compiler
                    "ld", "lds", "pc",
                    // protocol/interface definition
                    "proto", "thrift", "graphql", "gql",
                    // config languages
                    "dhall",
                    // database
                    "sql",
                    // template files
                    "j2", "jinja", "jinja2", "hbs", "handlebars", "mustache",
                    "tmpl", "template", "erb", "ejs", "tpl",
                    // diagram/graph formats
                    "mmd", "mermaid", "dot", "gv", "puml", "plantuml", "pu", "d2",
                    // documentation formats
                    "org", "tex", "latex", "adoc", "asciidoc", "rst",
                    // common config/doc files
                    "toml", "yaml", "yml", "json", "jsonc", "md", "txt", "ini", "conf", "config",
                    // container/virtualization
                    "dockerfile", "containerfile", "vagrantfile",
                    // build scripts
                    "makefile", "gnumakefile", "justfile",
                ];
                extensions.extend(code_exts.iter().map(|s| s.to_string()));
            }
            "docs" => {
                let doc_exts = ["md", "txt", "rst", "adoc"];
                extensions.extend(doc_exts.iter().map(|s| s.to_string()));
            }
            _ => {
                // treat as literal extension
                extensions.push(ext.to_lowercase());
            }
        }
    }

    extensions
}

fn embed_chunks(
    embedder: &mut backbone::embedding::OnnxEmbedder,
    chunks: &[String],
    batch_size: usize,
) -> std::result::Result<Vec<Vec<f32>>, String> {
    if chunks.is_empty() {
        return Ok(Vec::new());
    }

    let batch_size = batch_size.max(1);
    let mut all_embeddings = Vec::with_capacity(chunks.len());

    for chunk_batch_start in (0..chunks.len()).step_by(batch_size) {
        let chunk_batch_end = (chunk_batch_start + batch_size).min(chunks.len());
        let chunk_batch = &chunks[chunk_batch_start..chunk_batch_end];

        tracing::debug!(
            "embedding chunk batch [{}-{})",
            chunk_batch_start,
            chunk_batch_end
        );

        let embeddings = embedder
            .embed(chunk_batch)
            .map_err(|e| format!("embedding failed: {e}"))?;

        if embeddings.len() != chunk_batch.len() {
            return Err(format!(
                "embedding count mismatch: chunks={}, embeddings={}",
                chunk_batch.len(),
                embeddings.len()
            ));
        }

        all_embeddings.extend(embeddings);
    }

    Ok(all_embeddings)
}

async fn process_embedding_result(
    result: EmbeddingResult,
    qdrant_client: &qdrant_client::Qdrant,
    collection_name: &str,
    bm25_index: &mut backbone::retrieval::bm25::Bm25Index,
    state_store: &mut backbone::storage::StateStore,
    ui: &crate::terminal_layout::TerminalUiHandle,
) -> Result<()> {
    use qdrant_client::qdrant::{PointStruct, UpsertPointsBuilder, Value};
    use uuid::Uuid;

    let EmbeddingResult { task, embeddings } = result;
    let EmbeddingTask {
        file_path,
        source_path,
        display_name,
        file_size_bytes,
        modified_at,
        file_hash,
        chunks,
        total_chunk_bytes: _, // used for adaptive batch sizing in worker
    } = task;

    let embeddings = match embeddings {
        Ok(embeddings) => embeddings,
        Err(err) => {
            tracing::error!("embedding failed for {:?}: {}", file_path, err);
            let _ = state_store.record_failure(
                &source_path,
                Some(file_hash),
                file_size_bytes,
                modified_at,
                format!("embedding failed: {err}"),
            );
            let _ = state_store.mark_failed(&source_path, err);
            advance_file_progress(ui, &display_name, file_size_bytes);
            return Ok(());
        }
    };

    if embeddings.len() != chunks.len() {
        tracing::error!(
            "embedding count mismatch for {:?}: chunks={}, embeddings={}",
            file_path,
            chunks.len(),
            embeddings.len()
        );
        let _ = state_store.record_failure(
            &source_path,
            Some(file_hash),
            file_size_bytes,
            modified_at,
            "embedding count mismatch".to_string(),
        );
        let _ = state_store.mark_failed(&source_path, "embedding count mismatch".to_string());
        advance_file_progress(ui, &display_name, file_size_bytes);
        return Ok(());
    }

    let mut all_points = Vec::with_capacity(chunks.len());
    for (chunk_text, embedding) in chunks.iter().zip(embeddings.into_iter()) {
        let chunk_id = Uuid::new_v4().to_string();
        let mut payload: HashMap<String, Value> = HashMap::new();
        payload.insert("text".to_string(), chunk_text.clone().into());
        payload.insert(
            "source".to_string(),
            file_path.to_string_lossy().to_string().into(),
        );

        let point = PointStruct::new(chunk_id.clone(), embedding, payload);
        all_points.push(point);

        let _ = bm25_index.add_document(&chunk_id, chunk_text);
    }

    tracing::debug!("created {} points for {:?}", all_points.len(), file_path);

    let mut retry_count = 0;
    let max_retries = 3;
    let mut success = false;

    while retry_count <= max_retries && !success {
        let upsert_request =
            UpsertPointsBuilder::new(collection_name.to_string(), all_points.clone()).build();
        match qdrant_client.upsert_points(upsert_request).await {
            Ok(_) => {
                tracing::debug!("successfully ingested file {:?}", file_path);
                let _ = state_store.mark_complete(&source_path, all_points.len());
                let _ = state_store.clear_failure(&source_path);
                advance_file_progress(ui, &display_name, file_size_bytes);
                success = true;
            }
            Err(e) => {
                if retry_count < max_retries {
                    let delay_ms = 100 * (2_u64.pow(retry_count));
                    tracing::warn!(
                        "upsert failed for {:?} (attempt {}/{}), retrying in {}ms: {}",
                        file_path,
                        retry_count + 1,
                        max_retries + 1,
                        delay_ms,
                        e
                    );
                    tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
                    retry_count += 1;
                } else {
                    tracing::error!(
                        "failed to upsert points for {:?} after {} attempts: {}",
                        file_path,
                        max_retries + 1,
                        e
                    );
                    let _ = state_store.record_failure(
                        &source_path,
                        Some(file_hash.clone()),
                        file_size_bytes,
                        modified_at,
                        format!("upsert failed: {e}"),
                    );
                    let _ = state_store.mark_failed(&source_path, e.to_string());
                    advance_file_progress(ui, &display_name, file_size_bytes);
                    retry_count += 1;
                }
            }
        }
    }

    Ok(())
}

fn advance_file_progress(
    ui: &crate::terminal_layout::TerminalUiHandle,
    display_name: &str,
    file_size_bytes: u64,
) {
    ui.set_current_file(display_name.to_string(), file_size_bytes);
    ui.set_files_message(display_name.to_string());
    ui.set_bytes_message(format_filesize_message(file_size_bytes));
    ui.inc_files(1);
    ui.inc_bytes(file_size_bytes);
}

async fn drain_embedding_results(
    result_rx: &std::sync::mpsc::Receiver<EmbeddingResult>,
    pending_tasks: &mut usize,
    qdrant_client: &qdrant_client::Qdrant,
    collection_name: &str,
    bm25_index: &mut backbone::retrieval::bm25::Bm25Index,
    state_store: &mut backbone::storage::StateStore,
    ui: &crate::terminal_layout::TerminalUiHandle,
) -> Result<()> {
    loop {
        match result_rx.try_recv() {
            Ok(result) => {
                *pending_tasks = pending_tasks.saturating_sub(1);
                process_embedding_result(
                    result,
                    qdrant_client,
                    collection_name,
                    bm25_index,
                    state_store,
                    ui,
                )
                .await?;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => break,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => break,
        }
    }

    Ok(())
}

async fn sync_documents(args: SyncCommand) -> Result<()> {
    use backbone::config::BbtConfig;
    use backbone::document::loader::DocumentLoader;
    use backbone::document::text::chunking::chunk_text;
    use backbone::embedding::{ensure_onnx_model, EmbeddingModelInfo, ExecutionProvider, OnnxEmbedder};
    use backbone::retrieval::bm25::{load_index, save_index};
    use backbone::storage::StateStore;
    use backbone::tracing::init_tracing;
    use qdrant_client::Qdrant;
    use std::fs;
    use std::sync::{mpsc, Arc, Mutex};
    use std::thread;

    let config = BbtConfig::from_env()?;
    let loader = DocumentLoader::with_max_size(config.max_file_size_bytes)?;
    let state_path = config.state_store_path.clone();
    if args.reset_state {
        if state_path.exists() {
            tracing::info!(
                "resetting state store at {}",
                state_path.to_string_lossy()
            );
            fs::remove_file(&state_path)?;
        }
        let wal_path = PathBuf::from(format!("{}-wal", state_path.to_string_lossy()));
        let shm_path = PathBuf::from(format!("{}-shm", state_path.to_string_lossy()));
        if wal_path.exists() {
            let _ = fs::remove_file(wal_path);
        }
        if shm_path.exists() {
            let _ = fs::remove_file(shm_path);
        }
    }
    let mut state_store = StateStore::new(&state_path)?;
    let incomplete = state_store.finalize_incomplete("interrupted")?;
    if incomplete > 0 {
        tracing::info!("marked {} in-progress files as failed", incomplete);
    }
    let extensions = expand_extensions(&args.ext);

    let terminal_ui = crate::terminal_layout::TerminalUi::new()
        .map_err(|e| anyhow!("failed to set up terminal ui: {}", e))?;
    let ui = terminal_ui.handle();
    let _guard = init_tracing("bbt", terminal_ui.log_sender())?;

    ui.set_scan_message("starting scan".to_string());

    let scan_scope_key = build_scan_scope_key(&args, &extensions);

    let scan_run = state_store.get_scan_run(&scan_scope_key)?;
    let (candidates, scan_totals, scan_run_id) = if let Some(run) = scan_run {
        if run.status == backbone::storage::ScanStatus::Complete {
            let summary = state_store.scan_entries_summary(&scan_scope_key, &run.run_id)?;
            if summary.files_seen == 0 && run.files_seen > 0 {
                tracing::warn!("scan cache empty; re-scanning");
                let scan_progress = state_store.start_scan(&scan_scope_key)?;
                let (candidates, scan_totals) = collect_candidates(
                    &args,
                    &extensions,
                    &ui,
                    &mut state_store,
                    &scan_scope_key,
                    &scan_progress,
                )?;
                state_store.finish_scan(&scan_scope_key, &scan_progress.run_id)?;
                ui.finish_scan();
                tracing::info!(
                    "scan complete: {} files, {} bytes",
                    scan_totals.files,
                    scan_totals.bytes
                );
                (candidates, scan_totals, scan_progress.run_id)
            } else {
                if summary.files_seen != run.files_seen
                    || summary.bytes_seen != run.bytes_seen
                {
                    state_store.update_scan_run_counts(
                        &scan_scope_key,
                        &run.run_id,
                        summary.files_seen,
                        summary.bytes_seen,
                    )?;
                    tracing::info!(
                        "scan totals reconciled from cached entries ({} files, {} bytes)",
                        summary.files_seen,
                        summary.bytes_seen
                    );
                }
                ui.set_scan_message(format!(
                    "{} files, {} bytes",
                    summary.files_seen, summary.bytes_seen
                ));
                tracing::info!(
                    "using cached scan with {} files and {} bytes",
                    summary.files_seen,
                    summary.bytes_seen
                );
                let entries = state_store.list_scan_entries(&scan_scope_key, &run.run_id)?;
                if summary.files_seen > 0 && entries.is_empty() {
                    tracing::warn!("scan cache empty; re-scanning");
                    let scan_progress = state_store.start_scan(&scan_scope_key)?;
                    let (candidates, scan_totals) = collect_candidates(
                        &args,
                        &extensions,
                        &ui,
                        &mut state_store,
                        &scan_scope_key,
                        &scan_progress,
                    )?;
                    state_store.finish_scan(&scan_scope_key, &scan_progress.run_id)?;
                    ui.finish_scan();
                    tracing::info!(
                        "scan complete: {} files, {} bytes",
                        scan_totals.files,
                        scan_totals.bytes
                    );
                    (candidates, scan_totals, scan_progress.run_id)
                } else {
                    let candidates = entries
                        .into_iter()
                        .map(|entry| FileCandidate {
                            path: PathBuf::from(entry.source_path),
                            size_bytes: entry.file_size_bytes,
                            modified_at: entry.modified_at,
                        })
                        .collect::<Vec<_>>();
                    ui.finish_scan();
                    tracing::info!(
                        "scan cached: {} files, {} bytes",
                        summary.files_seen,
                        summary.bytes_seen
                    );
                    (
                        candidates,
                        ScanTotals {
                            files: summary.files_seen,
                            bytes: summary.bytes_seen,
                        },
                        run.run_id.clone(),
                    )
                }
            }
        } else {
            let scan_progress = state_store.start_scan(&scan_scope_key)?;
            ui.set_scan_message(format!(
                "{} files, {} bytes",
                scan_progress.files_seen, scan_progress.bytes_seen
            ));
            if scan_progress.files_seen > 0 {
                tracing::info!(
                    "resuming scan with {} files and {} bytes already counted",
                    scan_progress.files_seen,
                    scan_progress.bytes_seen
                );
            }
            let (candidates, scan_totals) = collect_candidates(
                &args,
                &extensions,
                &ui,
                &mut state_store,
                &scan_scope_key,
                &scan_progress,
            )?;
            state_store.finish_scan(&scan_scope_key, &scan_progress.run_id)?;
            ui.finish_scan();
            tracing::info!(
                "scan complete: {} files, {} bytes",
                scan_totals.files,
                scan_totals.bytes
            );
            (candidates, scan_totals, scan_progress.run_id)
        }
    } else {
        let scan_progress = state_store.start_scan(&scan_scope_key)?;
        ui.set_scan_message(format!(
            "{} files, {} bytes",
            scan_progress.files_seen, scan_progress.bytes_seen
        ));
        let (candidates, scan_totals) = collect_candidates(
            &args,
            &extensions,
            &ui,
            &mut state_store,
            &scan_scope_key,
            &scan_progress,
        )?;
        state_store.finish_scan(&scan_scope_key, &scan_progress.run_id)?;
        ui.finish_scan();
        tracing::info!(
            "scan complete: {} files, {} bytes",
            scan_totals.files,
            scan_totals.bytes
        );
        (candidates, scan_totals, scan_progress.run_id)
    };

    let total_files = scan_totals.files;
    let total_bytes = scan_totals.bytes;
    ui.set_files_total(total_files);
    ui.set_bytes_total(total_bytes);

    let mut resumed_files = 0_u64;
    let mut resumed_bytes = 0_u64;
    if !args.force {
        let summary =
            state_store.processed_summary_for_scan(&scan_scope_key, &scan_run_id)?;
        resumed_files = summary.files_seen.min(total_files);
        resumed_bytes = summary.bytes_seen.min(total_bytes);
    }

    if resumed_files > 0 {
        tracing::info!(
            "resuming with {} files and {} bytes already processed",
            resumed_files,
            resumed_bytes
        );
    }

    ui.set_files_position(resumed_files);
    ui.set_bytes_position(resumed_bytes);

    let model_info = EmbeddingModelInfo::default_model()
        .with_max_seq_len_cap(config.embedding_max_seq_len);
    let vector_dimensions = model_info.dimensions;

    let model_path = ensure_onnx_model(&model_info, &config.model_cache_dir)?;
    let embedding_provider = ExecutionProvider::detect();

    // load tokenizer once and share across all workers to avoid memory duplication
    let tokenizer_path = model_path.parent()
        .ok_or_else(|| anyhow::anyhow!("invalid model path"))?
        .join("tokenizer.json");
    let shared_tokenizer = std::sync::Arc::new(
        Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer from {:?}: {}", tokenizer_path, e))?
    );

    let embedding_workers = config.embedding_workers.max(1);
    let embedding_queue_size = config.embedding_queue_size.max(embedding_workers);
    let embedding_batch_size = config.embedding_batch_size.max(1);
    tracing::info!(
        "embedding workers={}, queue_size={}, batch_size={}",
        embedding_workers,
        embedding_queue_size,
        embedding_batch_size
    );

    // create pool of embedders for concurrent GPU access
    // more sessions = better GPU utilization but more VRAM (~1.5 GiB per session)
    // capped at 4 sessions to reduce VRAM pressure and provide each session more headroom
    //
    // to adjust session pool tolerance:
    //   - high VRAM (24+ GiB): use `embedding_workers / 2` for more parallelism
    //   - low VRAM (8 GiB):    use `(embedding_workers / 8).clamp(1, 2)` for fewer sessions
    //   - balanced (12-16 GiB): current formula works well
    let session_pool_size = (embedding_workers / 4).clamp(1, 4);
    tracing::info!(
        "initializing {} onnx sessions for concurrent GPU access",
        session_pool_size
    );

    let mut embedder_pool: Vec<Arc<Mutex<OnnxEmbedder>>> = Vec::with_capacity(session_pool_size);
    for i in 0..session_pool_size {
        let embedder = OnnxEmbedder::new_with_shared_tokenizer(
            &model_path,
            model_info.clone(),
            embedding_provider,
            Arc::clone(&shared_tokenizer),
            session_pool_size,
        )?;
        embedder_pool.push(Arc::new(Mutex::new(embedder)));
        tracing::info!("session {}/{} initialized", i + 1, session_pool_size);
    }
    tracing::info!(
        "{} sessions ready - {} workers will share them round-robin",
        session_pool_size,
        embedding_workers
    );

    // chunk budget: controls concurrent GPU memory pressure
    // - small files share the pool up to total budget
    // - large files (> threshold) are processed on CPU to avoid VRAM spikes
    let max_chunks_in_flight = session_pool_size * 64;
    let total_bytes_budget = args.max_pool_bytes_kib.min(150) * 1024; // convert KiB to bytes, max 150 KiB
    let large_file_threshold = args.large_file_threshold_kib * 1024; // convert KiB to bytes
    let chunk_budget = Arc::new(crate::chunk_budget::ChunkBudget::new(
        max_chunks_in_flight,
        total_bytes_budget,
        large_file_threshold,
    ));
    tracing::info!(
        "chunk budget: {} pool, files >{} processed on CPU",
        format_bytes_short(total_bytes_budget),
        format_bytes_short(large_file_threshold),
    );

    // create CPU embedder for large files to avoid VRAM pressure
    // large files are processed on CPU while small files use GPU pool
    let cpu_embedder: Option<Arc<Mutex<OnnxEmbedder>>> = if embedding_provider != ExecutionProvider::Cpu {
        tracing::info!("initializing CPU embedder for large file processing");
        match OnnxEmbedder::new_with_shared_tokenizer(
            &model_path,
            model_info.clone(),
            ExecutionProvider::Cpu,
            Arc::clone(&shared_tokenizer),
            1, // single CPU session
        ) {
            Ok(embedder) => {
                tracing::info!("CPU embedder ready for large files (>{} threshold)", format_bytes_short(large_file_threshold));
                Some(Arc::new(Mutex::new(embedder)))
            }
            Err(e) => {
                tracing::warn!("failed to create CPU embedder: {}, large files will use GPU", e);
                None
            }
        }
    } else {
        // already using CPU, no fallback needed
        None
    };

    // track active sessions for monitoring
    let active_sessions = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let total_sessions = session_pool_size;

    let (task_tx, task_rx) = mpsc::sync_channel::<EmbeddingTask>(embedding_queue_size);
    let (result_tx, result_rx) = mpsc::sync_channel::<EmbeddingResult>(embedding_queue_size);
    let task_rx = Arc::new(Mutex::new(task_rx));
    let mut worker_handles = Vec::new();

    for worker_id in 0..embedding_workers {
        let task_rx = Arc::clone(&task_rx);
        let result_tx = result_tx.clone();
        // assign worker to session round-robin
        let session_id = worker_id % session_pool_size;
        let gpu_embedder = Arc::clone(&embedder_pool[session_id]);
        let cpu_embedder = cpu_embedder.clone();
        let worker_batch_size = embedding_batch_size;
        let chunk_budget = Arc::clone(&chunk_budget);
        let active_sessions = Arc::clone(&active_sessions);

        let handle = thread::Builder::new()
            .name(format!("embedding_worker_{}", worker_id))
            .spawn(move || {
                tracing::info!("worker {} started (session {}, cpu fallback: {})",
                    worker_id, session_id, cpu_embedder.is_some());

                loop {
                    let task = {
                        let receiver = task_rx.lock().expect("task_rx lock poisoned");
                        receiver.recv()
                    };

                    let task = match task {
                        Ok(task) => task,
                        Err(_) => break,
                    };

                    let num_chunks = task.chunks.len();
                    let is_large_file = task.total_chunk_bytes > chunk_budget.large_file_threshold();

                    // acquire chunk budget before processing (blocks if over capacity)
                    // large files are processed on CPU to avoid VRAM pressure
                    let size_kib = task.total_chunk_bytes / 1024;
                    let use_cpu = is_large_file && cpu_embedder.is_some();

                    if is_large_file {
                        if use_cpu {
                            tracing::info!(
                                "large file ({} KiB, {} chunks) - processing on CPU",
                                size_kib, num_chunks
                            );
                        } else {
                            tracing::warn!(
                                "large file ({} KiB, {} chunks) - waiting for queue to drain (no CPU fallback)",
                                size_kib, num_chunks
                            );
                        }
                    }

                    // for CPU processing, skip budget acquisition since it doesn't use GPU
                    let _budget_guard = if !use_cpu {
                        Some(chunk_budget.acquire(num_chunks, task.total_chunk_bytes))
                    } else {
                        None
                    };

                    if is_large_file && !use_cpu {
                        tracing::warn!(
                            "large file ({} KiB, {} chunks) - now running exclusively on GPU",
                            size_kib, num_chunks
                        );
                    }

                    // adaptive batch size: larger files get larger batches
                    // for CPU: use smaller batches to reduce memory pressure
                    // for GPU: scale with file size
                    let adaptive_batch_size = if use_cpu {
                        // CPU: use smaller batches for better memory efficiency
                        worker_batch_size.min(16)
                    } else if !task.chunks.is_empty() && task.total_chunk_bytes > 0 {
                        // GPU: scale batch size with file size:
                        // - small files (<64 KiB): min batch (32)
                        // - medium files (64 KiB - 1 MiB): scale 32-128
                        // - large files (>1 MiB): max batch (256)
                        let file_kb = task.total_chunk_bytes / 1024;
                        let batch_size = if file_kb < 64 {
                            32
                        } else if file_kb < 1024 {
                            // linear scale from 32 to 128 between 64KB and 1MB
                            let ratio = (file_kb - 64) as f32 / (1024 - 64) as f32;
                            32 + (ratio * 96.0) as usize
                        } else {
                            // large files get max batch for maximum GPU utilization
                            worker_batch_size.min(256)
                        };
                        tracing::debug!(
                            worker = worker_id,
                            chunks = num_chunks,
                            file_kb,
                            batch_size,
                            "adaptive batch: larger files get bigger batches"
                        );
                        batch_size.clamp(16, worker_batch_size)
                    } else {
                        worker_batch_size
                    };

                    // use CPU embedder for large files, GPU for small files
                    let embeddings = if use_cpu {
                        let cpu_emb = cpu_embedder.as_ref().unwrap();
                        let mut embedder = cpu_emb.lock().expect("cpu_embedder lock poisoned");
                        embed_chunks(&mut *embedder, &task.chunks, adaptive_batch_size)
                    } else {
                        let mut embedder = gpu_embedder.lock().expect("embedder lock poisoned");
                        active_sessions.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let result = embed_chunks(&mut *embedder, &task.chunks, adaptive_batch_size);
                        active_sessions.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                        result
                    };
                    // _budget_guard dropped here, releasing chunk capacity

                    if result_tx
                        .send(EmbeddingResult {
                            task,
                            embeddings,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            })?;
        worker_handles.push(handle);
    }

    drop(result_tx);

    // load or create bm25 index
    tracing::info!("loading bm25 index from {:?}", config.bm25_index_path);
    let mut bm25_index = load_index(&config.bm25_index_path)?;
    tracing::info!("bm25 index loaded with {} documents", bm25_index.num_docs());

    // use rest client instead of grpc
    let qdrant_client = if let Some(api_key) = &config.qdrant_api_key {
        Qdrant::from_url(&config.qdrant_url)
            .api_key(api_key.clone())
            .timeout(std::time::Duration::from_secs(60))
            .build()?
    } else {
        Qdrant::from_url(&config.qdrant_url)
            .timeout(std::time::Duration::from_secs(60))
            .build()?
    };

    let collection_name = "bbt".to_string();

    // note: health_check() hangs with grpc client, skipping explicit health check
    // connection will be verified when we actually use the client
    tracing::info!("connecting to qdrant at {}", config.qdrant_url);

    // check if collection exists, create if needed
    use qdrant_client::qdrant::{CreateCollectionBuilder, Distance, VectorParamsBuilder};
    let collection_exists = qdrant_client
        .collection_exists(&collection_name)
        .await
        .unwrap_or(false);

    if !collection_exists {
        tracing::info!("creating collection '{}'", collection_name);
        let vector_params = VectorParamsBuilder::new(vector_dimensions as u64, Distance::Cosine).build();
        qdrant_client
            .create_collection(CreateCollectionBuilder::new(&collection_name).vectors_config(vector_params))
            .await?;
        tracing::info!("collection '{}' created", collection_name);
    } else {
        tracing::info!("using existing collection '{}'", collection_name);
    }
    // create shared resource stats for profiling
    #[cfg(feature = "heap-profiling")]
    let shared_stats = std::sync::Arc::new(std::sync::Mutex::new(
        crate::resource_monitor::ResourceStats::default()
    ));

    // flag to delay resource monitor display until sync starts
    let display_stats = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let display_stats_clone = Arc::clone(&display_stats);

    // spawn resource monitor update task
    let ui_clone = ui.clone();
    let chunk_budget_clone = Arc::clone(&chunk_budget);
    let active_sessions_clone = Arc::clone(&active_sessions);
    let resource_done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let resource_done_clone = Arc::clone(&resource_done);
    #[cfg(feature = "heap-profiling")]
    let shared_stats_clone = std::sync::Arc::clone(&shared_stats);

    let resource_update_handle = std::thread::spawn(move || {
        let mut monitor = crate::resource_monitor::ResourceMonitor::new();
        loop {
            // wait until sync has started before displaying stats
            if !display_stats_clone.load(std::sync::atomic::Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(100));
                continue;
            }

            let stats = monitor.get_stats();
            let active = active_sessions_clone.load(std::sync::atomic::Ordering::Relaxed);
            let chunks_in_flight = chunk_budget_clone.chunks_in_flight();
            let bytes_in_flight = chunk_budget_clone.bytes_in_flight();
            let bytes_limit = chunk_budget_clone.total_bytes_budget();
            let large_file = chunk_budget_clone.large_file_running();
            // compact format with clear labels:
            // cpu:820% gpu:12% mem:9.2g vram:1.5g sess:5/8 chunks:61 pool:15/128k [EXCL]
            let pool_used_kib = bytes_in_flight / 1024;
            let pool_limit_kib = bytes_limit / 1024;
            let exclusive_tag = if large_file { "[EXCL]" } else { "      " }; // 6 chars fixed
            let combined = format!(
                "{} sess:{:>2}/{:<2} chunks:{:>4} pool:{:>3}/{}k {}",
                stats.format_combined(),
                active,
                total_sessions,
                chunks_in_flight,
                pool_used_kib,
                pool_limit_kib,
                exclusive_tag,
            );
            ui_clone.set_resource_message(combined);

            // update shared stats for profiling
            #[cfg(feature = "heap-profiling")]
            if let Ok(mut guard) = shared_stats_clone.lock() {
                *guard = stats.clone();
            }

            std::thread::sleep(std::time::Duration::from_millis(200));

            // check if bar is finished (indicates main thread completed)
            if resource_done_clone.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
        }
        monitor.stop();
    });

    let mut pending_tasks = 0usize;

    #[cfg(feature = "heap-profiling")]
    let profiler_snapshot_interval = 50u64; // minimum file interval

    #[cfg(feature = "heap-profiling")]
    let mut last_snapshot_files = 0u64;

    #[cfg(feature = "heap-profiling")]
    let mut last_snapshot_memory_pct = 0.0f32;

    #[cfg(feature = "heap-profiling")]
    let memory_snapshot_thresholds = [25.0, 50.0, 75.0, 90.0, 95.0]; // memory % thresholds

    // log sync start and enable resource monitor display
    tracing::info!(
        "starting sync for paths: {:?}, extensions: {:?}, git_only={}, honor_gitignore={}",
        args.paths,
        extensions,
        args.git_only,
        args.honor_gitignore
    );
    display_stats.store(true, std::sync::atomic::Ordering::Relaxed);

    for candidate in candidates {
        drain_embedding_results(
            &result_rx,
            &mut pending_tasks,
            &qdrant_client,
            &collection_name,
            &mut bm25_index,
            &mut state_store,
            &ui,
        )
        .await?;

        let filename = candidate
            .path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or("unknown")
            .to_string();
        let file_path = candidate.path.as_path();
        let file_size_bytes = candidate.size_bytes;
        let modified_at = candidate.modified_at;
        let source_path = file_path.to_string_lossy().to_string();

        // detect git repository for display
        let repo_name = find_git_repo(file_path)
            .map(|root| get_repo_name(&root))
            .unwrap_or_else(|| "no-repo".to_string());
        let display_name = format!("{}/{}", repo_name, filename);
        ui.set_current_file(display_name.clone(), file_size_bytes);
        ui.set_files_message(display_name.clone());
        ui.set_bytes_message(format_filesize_message(file_size_bytes));

        // skip empty files silently (e.g., __init__.py)
        if file_size_bytes == 0 {
            tracing::debug!("skipping empty file {:?}", file_path);
            advance_file_progress(&ui, &display_name, file_size_bytes);
            continue;
        }

        let skip_existing = !args.force
            && (state_store.should_skip(
                &source_path,
                file_size_bytes,
                modified_at,
            ) || state_store.should_skip_failure(
                &source_path,
                file_size_bytes,
                modified_at,
            ));
        if skip_existing {
            tracing::debug!("skipping unchanged file {:?}", file_path);
            continue;
        }

        tracing::debug!("processing file: {:?}", file_path);

        let doc = match loader.load(file_path) {
            Ok(d) => d,
            Err(e) => {
                tracing::error!("failed to load file {:?}: {}", file_path, e);
                let _ = state_store.record_failure(
                    &source_path,
                    None,
                    file_size_bytes,
                    modified_at,
                    format!("load failed: {e}"),
                );
                let _ = state_store.mark_failed(&source_path, e.to_string());
                advance_file_progress(&ui, &display_name, file_size_bytes);
                continue;
            }
        };

        if let Err(e) = state_store.mark_processing(
            doc.file_hash.clone(),
            source_path.clone(),
            file_size_bytes,
            modified_at,
        ) {
            tracing::warn!("failed to update state for {:?}: {}", file_path, e);
        }

        let chunks = match chunk_text(&doc.content, config.chunk_size, config.chunk_overlap) {
            Ok(c) => c,
            Err(e) => {
                tracing::error!("failed to chunk file {:?}: {}", file_path, e);
                let _ = state_store.record_failure(
                    &source_path,
                    Some(doc.file_hash.clone()),
                    file_size_bytes,
                    modified_at,
                    format!("chunk failed: {e}"),
                );
                let _ = state_store.mark_failed(&source_path, e.to_string());
                advance_file_progress(&ui, &display_name, file_size_bytes);
                continue;
            }
        };

        if chunks.is_empty() {
            // files with only whitespace or very small content produce no chunks
            // this is not an error, just skip them silently
            tracing::debug!(
                "skipping file {:?} (no chunks produced - likely only whitespace)",
                file_path
            );
            advance_file_progress(&ui, &display_name, file_size_bytes);
            continue;
        }

        // calculate total chunk bytes for adaptive batch sizing
        let total_chunk_bytes: usize = chunks.iter().map(|c| c.len()).sum();

        let mut task = EmbeddingTask {
            file_path: file_path.to_path_buf(),
            source_path: source_path.clone(),
            display_name: display_name.clone(),
            file_size_bytes,
            modified_at,
            file_hash: doc.file_hash.clone(),
            chunks,
            total_chunk_bytes,
        };

        loop {
            match task_tx.try_send(task) {
                Ok(()) => {
                    pending_tasks += 1;

                    #[cfg(feature = "heap-profiling")]
                    {
                        let current_files = ui.files_position();
                        let stats = shared_stats.lock().unwrap().clone();

                        // calculate memory usage percentage
                        let memory_pct = if stats.memory_total_mb > 0 {
                            (stats.memory_used_mb as f32 / stats.memory_total_mb as f32) * 100.0
                        } else {
                            0.0
                        };

                        // determine if we should snapshot
                        let should_snapshot = if current_files != last_snapshot_files {
                            // interval-based: every N files
                            let interval_trigger = current_files > 0 && current_files % profiler_snapshot_interval == 0;

                            // memory-based: crossed a threshold since last snapshot
                            let memory_trigger = memory_snapshot_thresholds.iter().any(|&threshold| {
                                memory_pct >= threshold && last_snapshot_memory_pct < threshold
                            });

                            interval_trigger || memory_trigger
                        } else {
                            false
                        };

                        if should_snapshot {
                            // drop and recreate profiler to write snapshot
                            if let Ok(mut guard) = crate::profiler::PROFILER.lock() {
                                let snapshot_reason = if memory_pct >= 90.0 {
                                    "high memory pressure"
                                } else if memory_pct >= 75.0 {
                                    "elevated memory"
                                } else if memory_pct >= 50.0 {
                                    "moderate memory"
                                } else if current_files % profiler_snapshot_interval == 0 {
                                    "file interval"
                                } else {
                                    "memory threshold"
                                };

                                tracing::warn!(
                                    files_processed = current_files,
                                    memory_used_mb = stats.memory_used_mb,
                                    memory_total_mb = stats.memory_total_mb,
                                    memory_pct = format!("{:.1}%", memory_pct),
                                    reason = snapshot_reason,
                                    "writing heap snapshot"
                                );

                                // drop profiler (writes dhat-heap.json and prints summary to stderr)
                                guard.take();

                                // rename snapshot with timestamp and memory info
                                let timestamp = std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap()
                                    .as_secs();
                                let snapshot_name = format!(
                                    "dhat-heap-{}-files-{:.0}pct-{}.json",
                                    current_files,
                                    memory_pct,
                                    timestamp
                                );

                                if let Err(e) = std::fs::rename("dhat-heap.json", &snapshot_name) {
                                    tracing::warn!("failed to rename snapshot: {}", e);
                                } else {
                                    tracing::info!("heap snapshot: {} (mem: {:.1}%)", snapshot_name, memory_pct);
                                }

                                // create new profiler to continue tracking
                                *guard = Some(dhat::Profiler::new_heap());

                                last_snapshot_files = current_files;
                                last_snapshot_memory_pct = memory_pct;
                            }
                        }
                    }

                    // periodic bm25 index flushing for data safety
                    let current_files = ui.files_position();
                    const BM25_FLUSH_INTERVAL: u64 = 100;
                    if current_files > 0 && current_files % BM25_FLUSH_INTERVAL == 0 {
                        tracing::info!(
                            "periodic bm25 flush: {} docs at {} files",
                            bm25_index.num_docs(),
                            current_files
                        );
                        if let Err(e) = save_index(&bm25_index, &config.bm25_index_path) {
                            tracing::warn!("periodic bm25 flush failed: {}", e);
                        }
                    }

                    break;
                }
                Err(mpsc::TrySendError::Full(returned_task)) => {
                    task = returned_task;
                    drain_embedding_results(
                        &result_rx,
                        &mut pending_tasks,
                        &qdrant_client,
                        &collection_name,
                        &mut bm25_index,
                        &mut state_store,
                        &ui,
                    )
                    .await?;
                }
                Err(mpsc::TrySendError::Disconnected(_)) => {
                    return Err(anyhow!("embedding worker channel closed"));
                }
            }
        }

        drain_embedding_results(
            &result_rx,
            &mut pending_tasks,
            &qdrant_client,
            &collection_name,
            &mut bm25_index,
            &mut state_store,
            &ui,
        )
        .await?;
    }

    drop(task_tx);

    while pending_tasks > 0 {
        match result_rx.recv() {
            Ok(result) => {
                pending_tasks = pending_tasks.saturating_sub(1);
                process_embedding_result(
                    result,
                    &qdrant_client,
                    &collection_name,
                    &mut bm25_index,
                    &mut state_store,
                    &ui,
                )
                .await?;
            }
            Err(_) => break,
        }
    }

    for handle in worker_handles {
        let _ = handle.join();
    }

    ui.finish_files("done");
    ui.finish_bytes("done");
    resource_done.store(true, std::sync::atomic::Ordering::Relaxed);

    // wait for resource monitor thread to finish
    let _ = resource_update_handle.join();

    // save bm25 index
    tracing::info!("saving bm25 index with {} documents", bm25_index.num_docs());
    if let Err(e) = save_index(&bm25_index, &config.bm25_index_path) {
        tracing::error!("failed to save bm25 index: {}", e);
    }

    // index git commits if requested
    if args.commits {
        tracing::info!("indexing git commits");
        let mut commit_embedder =
            OnnxEmbedder::new(&model_path, model_info.clone(), embedding_provider)?;
        if let Err(e) = sync_commits(&args, &config, &qdrant_client, &mut commit_embedder).await {
            tracing::error!("failed to index commits: {}", e);
        }
    }

    tracing::info!("sync completed");

    // terminal ui cleanup is handled by Drop

    // write final heap snapshot after all progress bars are closed
    #[cfg(feature = "heap-profiling")]
    {
        if let Ok(mut guard) = crate::profiler::PROFILER.lock() {
            guard.take(); // drops profiler, writes dhat-heap.json and prints to stderr
        }
    }

    Ok(())
}

async fn sync_commits(
    args: &SyncCommand,
    config: &backbone::config::BbtConfig,
    qdrant_client: &qdrant_client::Qdrant,
    embedder: &mut backbone::embedding::OnnxEmbedder,
) -> Result<()> {
    use backbone::git::{CommitExtractor, chunk_commit, CommitChunkingStrategy};
    use chrono::NaiveDate;
    use qdrant_client::qdrant::{CreateCollectionBuilder, Distance, PointStruct, UpsertPointsBuilder, VectorParamsBuilder};
    use std::collections::HashMap;

    let collection_name = "bbt_commits";
    let vector_dimensions = config.embedding_dims;

    // create commits collection if it doesn't exist
    let collection_exists = qdrant_client
        .collection_exists(collection_name)
        .await
        .unwrap_or(false);

    if !collection_exists {
        tracing::info!("creating commits collection '{}'", collection_name);
        let vector_params = VectorParamsBuilder::new(vector_dimensions as u64, Distance::Cosine).build();
        qdrant_client
            .create_collection(CreateCollectionBuilder::new(collection_name).vectors_config(vector_params))
            .await?;
        tracing::info!("commits collection '{}' created", collection_name);
    }

    // parse commits_since date if provided
    let since = if let Some(date_str) = &args.commits_since {
        match NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
            Ok(date) => Some(date.and_hms_opt(0, 0, 0).unwrap().and_utc()),
            Err(e) => {
                tracing::warn!("failed to parse commits_since date '{}': {}", date_str, e);
                None
            }
        }
    } else {
        None
    };

    // discover git repositories in sync paths
    let mut repo_paths = Vec::new();
    for path in &args.paths {
        if path.is_dir() {
            // recursively find all repos in directory (max depth 10)
            let repos = find_git_repos_recursive(path, 10);
            for repo_path in repos {
                if !repo_paths.contains(&repo_path) {
                    repo_paths.push(repo_path.clone());
                }
            }
        } else {
            // for files, find the containing repo
            if let Some(repo_path) = find_git_repo(path) {
                if !repo_paths.contains(&repo_path) {
                    repo_paths.push(repo_path);
                }
            }
        }
    }

    tracing::info!("found {} git repositories for commit indexing", repo_paths.len());

    let mut total_commits = 0;
    let mut total_chunks = 0;
    let repo_count = repo_paths.len();

    // process each repository
    for repo_path in repo_paths {
        let repo_name = get_repo_name(&repo_path);
        tracing::info!("processing repository: {}", repo_name);

        // extract commits
        let extractor = match CommitExtractor::new(&repo_path) {
            Ok(e) => e,
            Err(err) => {
                tracing::warn!("failed to open repository {:?}: {}", repo_path, err);
                continue;
            }
        };

        let commits = match extractor.extract_commits(since, Some(args.max_commits)) {
            Ok(c) => c,
            Err(err) => {
                tracing::error!("failed to extract commits from {:?}: {}", repo_path, err);
                continue;
            }
        };

        tracing::info!("extracted {} commits from {}", commits.len(), repo_name);
        total_commits += commits.len();

        // process each commit
        for commit in commits {
            // chunk commit (message only for now)
            let chunks = match chunk_commit(&commit, CommitChunkingStrategy::MessageOnly, config.chunk_size, config.chunk_overlap) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!("failed to chunk commit {}: {}", commit.commit_hash, e);
                    continue;
                }
            };

            total_chunks += chunks.len();

            // extract chunk texts
            let chunk_texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();

            // generate embeddings
            let embeddings = match embedder.embed(&chunk_texts) {
                Ok(e) => e,
                Err(err) => {
                    tracing::error!("failed to embed chunks for commit {}: {}", commit.commit_hash, err);
                    continue;
                }
            };

            // debug: trace embedding values
            if !embeddings.is_empty() {
                let first_emb = &embeddings[0];
                let sum: f32 = first_emb.iter().sum();
                let first_5: Vec<f32> = first_emb.iter().take(5).copied().collect();
                tracing::debug!(
                    "sync_commits: embeddings count={}, first vector: len={}, sum={}, first_5={:?}",
                    embeddings.len(),
                    first_emb.len(),
                    sum,
                    first_5
                );
            }

            // create qdrant points
            let mut points = Vec::new();
            for (chunk, embedding) in chunks.iter().zip(embeddings.iter()) {
                let mut payload = HashMap::new();
                payload.insert("text".to_string(), chunk.content.clone().into());
                payload.insert("commit_hash".to_string(), commit.commit_hash.clone().into());
                payload.insert("repository_name".to_string(), commit.metadata.repository_name.clone().into());
                payload.insert("repository_path".to_string(), commit.repository_path.to_string_lossy().to_string().into());
                payload.insert("chunk_type".to_string(), chunk.metadata.chunk_type.as_str().to_string().into());
                payload.insert("chunk_index".to_string(), (chunk.metadata.chunk_index as i64).into());
                payload.insert("total_chunks".to_string(), (chunk.metadata.total_chunks as i64).into());
                payload.insert("author_name".to_string(), commit.metadata.author.name.clone().into());
                payload.insert("author_email".to_string(), commit.metadata.author.email.clone().into());
                payload.insert("committer_name".to_string(), commit.metadata.committer.name.clone().into());
                payload.insert("committer_email".to_string(), commit.metadata.committer.email.clone().into());
                payload.insert("commit_time".to_string(), commit.metadata.commit_time.to_rfc3339().into());
                payload.insert("classification".to_string(), commit.metadata.classification.as_str().to_string().into());
                payload.insert("files_changed_count".to_string(), (commit.metadata.diff_stats.files_changed as i64).into());
                payload.insert("insertions".to_string(), (commit.metadata.diff_stats.insertions as i64).into());
                payload.insert("deletions".to_string(), (commit.metadata.diff_stats.deletions as i64).into());

                // store file paths as array
                let file_paths_values: Vec<qdrant_client::qdrant::Value> = commit.metadata.diff_stats.file_paths
                    .iter()
                    .map(|path| path.clone().into())
                    .collect();
                payload.insert("file_paths".to_string(), qdrant_client::qdrant::Value {
                    kind: Some(qdrant_client::qdrant::value::Kind::ListValue(
                        qdrant_client::qdrant::ListValue { values: file_paths_values }
                    ))
                }.into());

                let point = PointStruct::new(
                    chunk.id.clone(),
                    embedding.clone(),
                    payload,
                );
                points.push(point);
            }

            // upsert to qdrant
            let upsert_request = UpsertPointsBuilder::new(collection_name.to_string(), points).build();
            if let Err(e) = qdrant_client.upsert_points(upsert_request).await {
                tracing::error!("failed to upsert commit {}: {}", commit.commit_hash, e);
            }
        }
    }

    tracing::info!(
        "indexed {} commits ({} chunks) from {} repositories",
        total_commits,
        total_chunks,
        repo_count
    );

    Ok(())
}

struct FileCandidate {
    path: PathBuf,
    size_bytes: u64,
    modified_at: Option<chrono::DateTime<chrono::Utc>>,
}

enum ScanEvent {
    File(FileCandidate),
    Error(String),
}

struct ScanTotals {
    files: u64,
    bytes: u64,
}

fn collect_candidates(
    args: &SyncCommand,
    extensions: &[String],
    ui: &crate::terminal_layout::TerminalUiHandle,
    state_store: &mut backbone::storage::StateStore,
    scan_scope_key: &str,
    scan_progress: &backbone::storage::ScanProgress,
) -> Result<(Vec<FileCandidate>, ScanTotals)> {
    use ignore::{WalkBuilder, WalkState};
    use std::collections::HashSet;
    use std::sync::{mpsc, Arc};
    use std::thread;

    let mut candidates = Vec::new();
    let mut totals = ScanTotals {
        files: scan_progress.files_seen,
        bytes: scan_progress.bytes_seen,
    };

    let mut roots = Vec::new();
    for path in &args.paths {
        if !path.is_dir() {
            tracing::warn!("path {:?} is not a directory, skipping", path);
            continue;
        }
        roots.push(path.clone());
    }

    if roots.is_empty() {
        return Ok((candidates, totals));
    }

    let extensions_set: HashSet<String> = extensions.iter().cloned().collect();
    let extensions_set = Arc::new(extensions_set);

    let (tx, rx) = mpsc::channel::<ScanEvent>();
    let worker_tx = tx.clone();
    let worker_roots = roots.clone();
    let worker_extensions = Arc::clone(&extensions_set);
    let git_only = args.git_only;
    let honor_gitignore = args.honor_gitignore;

    let worker = thread::spawn(move || {
        let mut builder = WalkBuilder::new(&worker_roots[0]);
        for path in worker_roots.iter().skip(1) {
            builder.add(path);
        }

        builder.hidden(false).ignore(false);
        if honor_gitignore {
            builder.git_ignore(true).git_global(true).git_exclude(true);
        } else {
            builder.git_ignore(false).git_global(false).git_exclude(false);
        }
        if git_only {
            builder.require_git(true);
        }

        // add .dockerignore support
        builder.add_custom_ignore_filename(".dockerignore");

        // filter out common directories that should never be scanned
        builder.filter_entry(|entry| {
            let file_name = entry.file_name().to_string_lossy();
            let path_str = entry.path().to_string_lossy();

            // always exclude these directories
            let excluded_dirs = [
                // python
                ".venv", "venv", "__pycache__", ".tox", ".pytest_cache",
                "site-packages", ".eggs", "*.egg-info",
                // node.js
                "node_modules", ".npm", ".yarn", ".pnp", ".pnp.js",
                // rust
                "target",
                // general build/cache
                ".cache", "build", "dist", ".next", ".nuxt", ".svelte-kit",
                // version control
                ".git", ".svn", ".hg",
                // ide/editor
                ".vscode", ".idea", ".vs", ".DS_Store",
                // docker/containers
                ".docker",
                // data/logs
                "data", "logs", "tmp", "temp",
            ];

            for excluded in &excluded_dirs {
                if file_name == *excluded || file_name.ends_with(excluded) {
                    return false;
                }
            }

            // skip if path contains any excluded directory
            for excluded in &excluded_dirs {
                if path_str.contains(&format!("/{}/", excluded)) ||
                   path_str.contains(&format!("\\{}/", excluded)) {
                    return false;
                }
            }

            true
        });

        let threads = std::thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(4);
        builder.threads(threads);

        let walker = builder.build_parallel();
        walker.run(|| {
            let sender = worker_tx.clone();
            let extensions = Arc::clone(&worker_extensions);
            Box::new(move |result| {
                let entry = match result {
                    Ok(entry) => entry,
                    Err(err) => {
                        let _ = sender.send(ScanEvent::Error(format!(
                            "scan entry error: {}",
                            err
                        )));
                        return WalkState::Continue;
                    }
                };

                if !entry
                    .file_type()
                    .map(|file_type| file_type.is_file())
                    .unwrap_or(false)
                {
                    return WalkState::Continue;
                }

                let file_path = entry.path();
                let ext = match file_path.extension().and_then(|value| value.to_str()) {
                    Some(ext) => ext.to_lowercase(),
                    None => return WalkState::Continue,
                };
                if !extensions.contains(&ext) {
                    return WalkState::Continue;
                }

                let metadata = match entry.metadata() {
                    Ok(metadata) => metadata,
                    Err(err) => {
                        let _ = sender.send(ScanEvent::Error(format!(
                            "failed to read metadata for {:?}: {}",
                            file_path, err
                        )));
                        return WalkState::Continue;
                    }
                };

                let size_bytes = metadata.len();
                let modified_at = metadata
                    .modified()
                    .ok()
                    .map(|time| chrono::DateTime::<chrono::Utc>::from(time));

                let candidate = FileCandidate {
                    path: file_path.to_path_buf(),
                    size_bytes,
                    modified_at,
                };

                if sender.send(ScanEvent::File(candidate)).is_err() {
                    return WalkState::Quit;
                }

                WalkState::Continue
            })
        });
    });

    drop(tx);

    for event in rx {
        match event {
            ScanEvent::File(candidate) => {
                let source_path = candidate.path.to_string_lossy().to_string();
                let progress = match state_store.record_scan_entry(
                    scan_scope_key,
                    &scan_progress.run_id,
                    &source_path,
                    candidate.size_bytes,
                    candidate.modified_at,
                ) {
                    Ok(progress) => progress,
                    Err(e) => {
                        tracing::warn!(
                            "failed to update scan state for {:?}: {}",
                            candidate.path,
                            e
                        );
                        backbone::storage::ScanProgress {
                            run_id: scan_progress.run_id.clone(),
                            files_seen: totals.files,
                            bytes_seen: totals.bytes,
                        }
                    }
                };

                totals.files = progress.files_seen;
                totals.bytes = progress.bytes_seen;
                ui.set_scan_message(format_scan_message(
                    totals.files,
                    totals.bytes,
                    &source_path,
                ));

                candidates.push(candidate);
            }
            ScanEvent::Error(message) => {
                tracing::warn!("{message}");
            }
        }
    }

    if worker.join().is_err() {
        return Err(anyhow::anyhow!("scan worker panicked"));
    }

    ui.set_scan_message(format!("{} files, {} bytes", totals.files, totals.bytes));

    Ok((candidates, totals))
}

fn build_scan_scope_key(args: &SyncCommand, extensions: &[String]) -> String {
    let mut paths: Vec<String> = args
        .paths
        .iter()
        .map(|path| path.to_string_lossy().to_string())
        .collect();
    paths.sort();

    let mut ext_list = extensions.to_vec();
    ext_list.sort();

    format!(
        "paths={};ext={};git_only={};honor_gitignore={}",
        paths.join("|"),
        ext_list.join("|"),
        args.git_only,
        args.honor_gitignore
    )
}

fn format_scan_message(
    files: u64,
    bytes: u64,
    source_path: &str,
) -> String {
    format!("{} files {} bytes {}", files, bytes, source_path)
}

/// find git repository root for a given path
fn find_git_repo(path: &Path) -> Option<PathBuf> {
    // start from the path itself if it's a directory, or its parent if it's a file
    let start = if path.is_dir() {
        Some(path)
    } else {
        path.parent()
    };

    let mut current = start;
    while let Some(dir) = current {
        let git_dir = dir.join(".git");
        if git_dir.exists() {
            return Some(dir.to_path_buf());
        }
        current = dir.parent();
    }
    None
}

/// recursively find all git repositories under a directory
fn find_git_repos_recursive(root: &Path, max_depth: usize) -> Vec<PathBuf> {
    use std::collections::HashSet;
    use walkdir::WalkDir;

    let mut repos = Vec::new();
    let mut seen = HashSet::new();

    for entry in WalkDir::new(root)
        .max_depth(max_depth)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            let file_name = e.file_name().to_string_lossy();

            // always allow .git directories
            if file_name == ".git" {
                return true;
            }

            // skip common directories that should never be scanned
            let excluded_dirs = [
                // python
                ".venv", "venv", "__pycache__", ".tox", ".pytest_cache",
                "site-packages", ".eggs",
                // node.js
                "node_modules", ".npm", ".yarn", ".pnp",
                // rust
                "target",
                // general build/cache
                ".cache", "build", "dist", ".next", ".nuxt", ".svelte-kit",
                // version control (except .git)
                ".svn", ".hg",
                // ide/editor
                ".vscode", ".idea", ".vs", ".DS_Store",
                // docker/containers
                ".docker",
                // data/logs
                "data", "logs", "tmp", "temp",
            ];

            for excluded in &excluded_dirs {
                if file_name == *excluded {
                    return false;
                }
            }

            true
        })
    {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                tracing::debug!("skipping entry: {}", e);
                continue;
            }
        };

        // check if this is a .git directory
        if entry.file_name() == ".git" && entry.file_type().is_dir() {
            if let Some(repo_root) = entry.path().parent() {
                let canonical = match repo_root.canonicalize() {
                    Ok(p) => p,
                    Err(_) => repo_root.to_path_buf(),
                };

                // deduplicate using canonical paths
                if seen.insert(canonical.clone()) {
                    repos.push(canonical);
                    tracing::debug!("found git repo: {}", repo_root.display());
                }
            }
        }
    }

    repos
}

/// extract repository name from git repo root
fn get_repo_name(repo_root: &Path) -> String {
    repo_root
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string()
}

fn format_filesize_message(bytes: u64) -> String {
    use indicatif::HumanBytes;
    let size_str = format!("{}", HumanBytes(bytes));
    size_str
}

/// Format bytes as fixed-width compact string (5 chars): " 1.2g", "  46k", " 512b"
fn format_bytes_short(bytes: usize) -> String {
    const KIB: usize = 1024;
    const MIB: usize = 1024 * 1024;
    const GIB: usize = 1024 * 1024 * 1024;

    let s = if bytes >= GIB {
        format!("{:.1}g", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.1}m", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{}k", bytes / KIB)
    } else {
        format!("{}b", bytes)
    };
    // fixed 5-char width, right-aligned
    format!("{:>5}", s)
}

async fn query_documents(args: QueryCommand) -> Result<()> {
    use backbone::config::{BbtConfig, RetrievalMode};
    use backbone::embedding::{ensure_onnx_model, EmbeddingModelInfo, OnnxEmbedder};
    use backbone::embedding::ExecutionProvider as EmbeddingExecutionProvider;
    use backbone::reranking::{ensure_rerank_model, ExecutionProvider as RerankExecutionProvider, OnnxReranker, RerankModelInfo};
    use backbone::retrieval::{Bm25Scorer, FusionStrategy, fuse_results};
    use backbone::retrieval::bm25::load_index;
    use backbone::tracing::init_tracing;
    use qdrant_client::qdrant::{GetPointsBuilder, SearchPointsBuilder};
    use qdrant_client::Qdrant;

    let _guard = init_tracing("bbt", None)?;
    let mut config = BbtConfig::from_env()?;

    // apply CLI overrides
    config.retrieval_mode = args.mode.into();
    config.top_k = args.top_k;

    // handle --json shorthand
    let output_format = if args.json { OutputFormat::Json } else { args.format };

    // force vector mode for commit queries (no bm25 index for commits)
    if args.commits && config.retrieval_mode != RetrievalMode::Vector {
        tracing::info!("forcing vector mode for commit query (bm25/hybrid not supported for commits)");
        config.retrieval_mode = RetrievalMode::Vector;
    }

    let query = args.query.trim();
    if query.is_empty() {
        anyhow::bail!("query is required");
    }

    // load bm25 index if needed
    let bm25_index = if config.retrieval_mode == RetrievalMode::Bm25
        || config.retrieval_mode == RetrievalMode::Hybrid
    {
        tracing::info!("loading bm25 index from {:?}", config.bm25_index_path);
        Some(load_index(&config.bm25_index_path)?)
    } else {
        None
    };

    // calculate initial retrieval limit
    // if reranking is enabled, retrieve more candidates for better recall
    let rerank_multiplier = if config.enable_rerank { 3 } else { 1 };
    let initial_limit = config.top_k * rerank_multiplier;

    // perform retrieval based on mode
    let candidates: Vec<QueryResultOutput> = match config.retrieval_mode {
        RetrievalMode::Vector => {
            tracing::info!("vector search with top_k={}", initial_limit);

            let model_info = EmbeddingModelInfo::default_model()
                .with_max_seq_len_cap(config.embedding_max_seq_len);
            let model_path = ensure_onnx_model(&model_info, &config.model_cache_dir)?;
            let mut embedder = OnnxEmbedder::new(&model_path, model_info, EmbeddingExecutionProvider::detect())?;

            let qdrant_client = if let Some(api_key) = &config.qdrant_api_key {
                Qdrant::from_url(&config.qdrant_url)
                    .api_key(api_key.clone())
                    .timeout(std::time::Duration::from_secs(60))
                    .build()?
            } else {
                Qdrant::from_url(&config.qdrant_url)
                    .timeout(std::time::Duration::from_secs(60))
                    .build()?
            };

            let embeddings = embedder.embed(&[query.to_string()])?;
            let query_vector = embeddings
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("failed to embed query"))?;

            // select collection based on query type
            let collection_name = if args.commits {
                "bbt_commits".to_string()
            } else {
                "bbt".to_string()
            };

            let collection_exists = qdrant_client
                .collection_exists(&collection_name)
                .await
                .unwrap_or(false);
            if !collection_exists {
                let hint = if args.commits {
                    "run sync --commits first"
                } else {
                    "run sync first"
                };
                anyhow::bail!("collection '{}' not found, {}", collection_name, hint);
            }

            let mut search_builder =
                SearchPointsBuilder::new(collection_name.clone(), query_vector, initial_limit as u64)
                    .with_payload(true);

            // apply filters
            if let Some(filter) = if args.commits {
                build_commit_filter(&args)?
            } else {
                build_query_filter(&args)
            } {
                search_builder = search_builder.filter(filter);
            }

            let response = qdrant_client.search_points(search_builder).await?;

            // convert qdrant results to intermediate format
            let mut results = Vec::new();
            for point in response.result {
                let payload = point.payload;
                let text = payload
                    .get("text")
                    .and_then(|value| value_as_string(value))
                    .unwrap_or_default();
                let source_raw = payload
                    .get("source")
                    .and_then(|value| value_as_string(value));
                let citation_raw = format_citation(source_raw.as_deref());
                let citation = display_citation(&citation_raw, source_raw.as_deref(), CitationMode::Full);
                let source = display_source(source_raw.as_deref(), CitationMode::Full);
                let id = point_id_to_string(point.id);

                // extract metadata (commit-specific or general)
                let mut metadata = std::collections::HashMap::new();
                if args.commits {
                    // extract commit metadata
                    if let Some(commit_hash) = payload.get("commit_hash").and_then(|v| value_as_string(v)) {
                        metadata.insert("commit_hash".to_string(), commit_hash);
                    }
                    if let Some(author_name) = payload.get("author_name").and_then(|v| value_as_string(v)) {
                        metadata.insert("author_name".to_string(), author_name);
                    }
                    if let Some(author_email) = payload.get("author_email").and_then(|v| value_as_string(v)) {
                        metadata.insert("author_email".to_string(), author_email);
                    }
                    if let Some(commit_time) = payload.get("commit_time").and_then(|v| value_as_string(v)) {
                        metadata.insert("commit_time".to_string(), commit_time);
                    }
                    if let Some(classification) = payload.get("classification").and_then(|v| value_as_string(v)) {
                        metadata.insert("classification".to_string(), classification);
                    }
                    if let Some(insertions) = payload.get("insertions").and_then(|v| value_as_string(v)) {
                        metadata.insert("insertions".to_string(), insertions);
                    }
                    if let Some(deletions) = payload.get("deletions").and_then(|v| value_as_string(v)) {
                        metadata.insert("deletions".to_string(), deletions);
                    }
                    if let Some(repository_name) = payload.get("repository_name").and_then(|v| value_as_string(v)) {
                        metadata.insert("repository_name".to_string(), repository_name);
                    }
                    // file_paths is an array, join to string
                    if let Some(files) = payload.get("file_paths") {
                        if let Some(qdrant_client::qdrant::Value { kind: Some(qdrant_client::qdrant::value::Kind::ListValue(list)) }) = Some(files) {
                            let file_list: Vec<String> = list
                                .values
                                .iter()
                                .filter_map(|v| value_as_string(v))
                                .collect();
                            if !file_list.is_empty() {
                                metadata.insert("file_paths".to_string(), file_list.join(", "));
                            }
                        }
                    }
                }

                results.push(QueryResultOutput {
                    id,
                    score: point.score,
                    text,
                    citation,
                    source,
                    metadata,
                });
            }

            // sort results
            if args.commits && args.date_sort {
                // sort commits by date (newest first)
                use chrono::Utc;
                results.sort_by(|a, b| {
                    let a_time = a.metadata.get("commit_time")
                        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                        .map(|dt| dt.with_timezone(&Utc));
                    let b_time = b.metadata.get("commit_time")
                        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                        .map(|dt| dt.with_timezone(&Utc));
                    b_time.cmp(&a_time)
                });
            } else {
                // default: sort by score descending
                results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            }

            results
        }
        RetrievalMode::Bm25 => {
            tracing::info!("bm25 search with top_k={}", initial_limit);

            let index = bm25_index.as_ref().unwrap();
            let scorer = Bm25Scorer::with_params(config.bm25_k1, config.bm25_b);
            let bm25_results = scorer.search(index, query, initial_limit);

            tracing::info!("bm25 returned {} results", bm25_results.len());

            if bm25_results.is_empty() {
                Vec::new()
            } else {
                // fetch text and source from qdrant for bm25 results
                let qdrant_client = if let Some(api_key) = &config.qdrant_api_key {
                    Qdrant::from_url(&config.qdrant_url)
                        .api_key(api_key.clone())
                        .timeout(std::time::Duration::from_secs(60))
                        .build()?
                } else {
                    Qdrant::from_url(&config.qdrant_url)
                        .timeout(std::time::Duration::from_secs(60))
                        .build()?
                };

                let collection_name = "bbt".to_string();
                let point_ids: Vec<qdrant_client::qdrant::PointId> = bm25_results
                    .iter()
                    .map(|r| qdrant_client::qdrant::PointId::from(r.doc_id.clone()))
                    .collect();

                let get_response = qdrant_client
                    .get_points(
                        GetPointsBuilder::new(collection_name, point_ids)
                            .with_payload(true),
                    )
                    .await?;

                // build lookup map from qdrant response
                let payload_map: std::collections::HashMap<String, (String, String)> = get_response
                    .result
                    .into_iter()
                    .map(|point| {
                        let id = point_id_to_string(point.id);
                        let text = point
                            .payload
                            .get("text")
                            .and_then(|v| value_as_string(v))
                            .unwrap_or_default();
                        let source = point
                            .payload
                            .get("source")
                            .and_then(|v| value_as_string(v))
                            .unwrap_or_default();
                        (id, (text, source))
                    })
                    .collect();

                // convert bm25 results with fetched text
                bm25_results
                    .into_iter()
                    .map(|result| {
                        let (text, source_path) = payload_map
                            .get(&result.doc_id)
                            .cloned()
                            .unwrap_or_default();
                        let source_raw = if source_path.is_empty() {
                            Some(result.doc_id.clone())
                        } else {
                            Some(source_path)
                        };
                        let citation_raw = format_citation(source_raw.as_deref());
                        let citation = display_citation(&citation_raw, source_raw.as_deref(), CitationMode::Full);
                        let source = display_source(source_raw.as_deref(), CitationMode::Full);

                        QueryResultOutput {
                            id: result.doc_id,
                            score: result.score,
                            text,
                            citation,
                            source,
                            metadata: std::collections::HashMap::new(),
                        }
                    })
                    .collect()
            }
        }
        RetrievalMode::Hybrid => {
            tracing::info!("hybrid search (vector + bm25) with top_k={}", initial_limit);

            // perform vector search
            let model_info = EmbeddingModelInfo::default_model()
                .with_max_seq_len_cap(config.embedding_max_seq_len);
            let model_path = ensure_onnx_model(&model_info, &config.model_cache_dir)?;
            let mut embedder = OnnxEmbedder::new(&model_path, model_info, EmbeddingExecutionProvider::detect())?;

            let qdrant_client = if let Some(api_key) = &config.qdrant_api_key {
                Qdrant::from_url(&config.qdrant_url)
                    .api_key(api_key.clone())
                    .timeout(std::time::Duration::from_secs(60))
                    .build()?
            } else {
                Qdrant::from_url(&config.qdrant_url)
                    .timeout(std::time::Duration::from_secs(60))
                    .build()?
            };

            let embeddings = embedder.embed(&[query.to_string()])?;
            let query_vector = embeddings
                .into_iter()
                .next()
                .ok_or_else(|| anyhow::anyhow!("failed to embed query"))?;

            // debug: show embedding provider and sample values
            if args.verbose {
                let norm: f32 = query_vector.iter().map(|x| x * x).sum::<f32>().sqrt();
                let first_5: Vec<f32> = query_vector.iter().take(5).cloned().collect();
                let last_5: Vec<f32> = query_vector.iter().rev().take(5).rev().cloned().collect();
                println!("\n=== debug: query embedding ===");
                println!("provider: {}", embedder.provider().as_str());
                println!("dimensions: {}", query_vector.len());
                println!("l2 norm: {:.6} (should be ~1.0 for normalized)", norm);
                println!("first 5 values: {:?}", first_5);
                println!("last 5 values: {:?}", last_5);
            }

            let collection_name = "bbt".to_string();
            let collection_exists = qdrant_client
                .collection_exists(&collection_name)
                .await
                .unwrap_or(false);
            if !collection_exists {
                anyhow::bail!("collection '{}' not found, run sync first", collection_name);
            }

            let mut search_builder =
                SearchPointsBuilder::new(collection_name.clone(), query_vector, initial_limit as u64 * 2)
                    .with_payload(true);

            if let Some(filter) = build_query_filter(&args) {
                search_builder = search_builder.filter(filter);
            }

            let response = qdrant_client.search_points(search_builder).await?;

            // convert vector results and build source map in single pass
            let mut vector_source_map: std::collections::HashMap<String, String> = std::collections::HashMap::new();
            let vector_results: Vec<(String, f32, String)> = response
                .result
                .into_iter()
                .map(|point| {
                    let id = point_id_to_string(point.id);
                    let text = point
                        .payload
                        .get("text")
                        .and_then(|value| value_as_string(value))
                        .unwrap_or_default();
                    let source = point
                        .payload
                        .get("source")
                        .and_then(|value| value_as_string(value))
                        .unwrap_or_default();
                    vector_source_map.insert(id.clone(), source);
                    (id, point.score, text)
                })
                .collect();

            // perform bm25 search
            let index = bm25_index.as_ref().unwrap();
            let scorer = Bm25Scorer::with_params(config.bm25_k1, config.bm25_b);
            let bm25_results = scorer.search(index, query, initial_limit * 2);

            tracing::info!(
                "hybrid: vector returned {}, bm25 returned {}",
                vector_results.len(),
                bm25_results.len()
            );

            // debug: show raw scores and overlap analysis
            if args.verbose {
                println!("\n=== debug: raw vector scores ===");
                for (i, (id, score, text)) in vector_results.iter().take(5).enumerate() {
                    let text_preview: String = text.chars().take(50).collect();
                    println!("  {}: score={:.4} id={} text=\"{}...\"", i + 1, score, &id[..8.min(id.len())], text_preview);
                }
                if vector_results.len() > 5 {
                    println!("  ... and {} more", vector_results.len() - 5);
                }

                println!("\n=== debug: raw bm25 scores ===");
                for (i, result) in bm25_results.iter().take(5).enumerate() {
                    println!("  {}: score={:.4} id={}", i + 1, result.score, &result.doc_id[..8.min(result.doc_id.len())]);
                }
                if bm25_results.len() > 5 {
                    println!("  ... and {} more", bm25_results.len() - 5);
                }

                // overlap analysis
                let vector_ids: std::collections::HashSet<&str> = vector_results.iter().map(|(id, _, _)| id.as_str()).collect();
                let bm25_ids: std::collections::HashSet<&str> = bm25_results.iter().map(|r| r.doc_id.as_str()).collect();
                let overlap: Vec<&str> = vector_ids.intersection(&bm25_ids).cloned().collect();
                println!("\n=== debug: overlap analysis ===");
                println!("  vector-only: {}", vector_results.len() - overlap.len());
                println!("  bm25-only: {}", bm25_results.len() - overlap.len());
                println!("  in both: {}", overlap.len());
                if !overlap.is_empty() {
                    println!("  overlapping ids: {:?}", overlap.iter().take(3).collect::<Vec<_>>());
                }

                // embedding verification: compare stored vs freshly generated
                if let Some((top_id, _, top_text)) = vector_results.first() {
                    if !top_text.is_empty() {
                        println!("\n=== debug: embedding verification ===");
                        // fetch stored vector
                        let point_ids = vec![qdrant_client::qdrant::PointId::from(top_id.clone())];
                        if let Ok(get_resp) = qdrant_client
                            .get_points(
                                GetPointsBuilder::new(collection_name.clone(), point_ids)
                                    .with_vectors(true)
                                    .with_payload(true),
                            )
                            .await
                        {
                            if let Some(point) = get_resp.result.first() {
                                // extract stored vector
                                if let Some(vectors) = &point.vectors {
                                    if let Some(qdrant_client::qdrant::vectors::VectorsOptions::Vector(v)) = &vectors.vectors_options {
                                        let stored_vec: Vec<f32> = v.data.clone();
                                        let stored_norm: f32 = stored_vec.iter().map(|x| x * x).sum::<f32>().sqrt();

                                        // re-embed the text
                                        if let Ok(fresh_embeddings) = embedder.embed(&[top_text.clone()]) {
                                            if let Some(fresh_vec) = fresh_embeddings.first() {
                                                let fresh_norm: f32 = fresh_vec.iter().map(|x| x * x).sum::<f32>().sqrt();

                                                // compute cosine similarity between stored and fresh
                                                let dot: f32 = stored_vec.iter().zip(fresh_vec.iter()).map(|(a, b)| a * b).sum();
                                                let cos_sim = dot / (stored_norm * fresh_norm);

                                                println!("  stored vector: {} dims, l2 norm={:.6}", stored_vec.len(), stored_norm);
                                                println!("  fresh vector: {} dims, l2 norm={:.6}", fresh_vec.len(), fresh_norm);
                                                println!("  cosine similarity (stored vs fresh): {:.6}", cos_sim);
                                                println!("  stored first 5: {:?}", &stored_vec[..5.min(stored_vec.len())]);
                                                println!("  fresh first 5: {:?}", &fresh_vec[..5.min(fresh_vec.len())]);

                                                if cos_sim < 0.99 {
                                                    println!("  WARNING: embeddings don't match! cosine < 0.99");
                                                    println!("  this suggests the embedding model or provider has changed");
                                                } else {
                                                    println!("  embeddings match (cosine >= 0.99)");
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // adaptive scoring: adjust weights based on vector confidence
            // when vector search returns low-confidence results, heavily favor BM25
            const LOW_CONFIDENCE_VECTOR_WEIGHT: f32 = 0.2;
            const LOW_CONFIDENCE_BM25_WEIGHT: f32 = 0.8;

            let (effective_vector_weight, effective_bm25_weight) = if config.adaptive_scoring {
                let top_vector_score = vector_results.first().map(|(_, s, _)| *s).unwrap_or(0.0);

                if top_vector_score < config.adaptive_threshold {
                    // low vector confidence → heavily favor BM25
                    tracing::info!(
                        "adaptive scoring: top_vector_score={:.4} < threshold={:.2}, using low-confidence weights ({:.2}, {:.2})",
                        top_vector_score,
                        config.adaptive_threshold,
                        LOW_CONFIDENCE_VECTOR_WEIGHT,
                        LOW_CONFIDENCE_BM25_WEIGHT
                    );
                    if args.verbose {
                        println!("\n=== debug: adaptive scoring ===");
                        println!("  top vector score: {:.4}", top_vector_score);
                        println!("  threshold: {:.2}", config.adaptive_threshold);
                        println!("  decision: LOW CONFIDENCE → favor BM25");
                        println!("  weights: vector={:.2}, bm25={:.2}", LOW_CONFIDENCE_VECTOR_WEIGHT, LOW_CONFIDENCE_BM25_WEIGHT);
                    }
                    (LOW_CONFIDENCE_VECTOR_WEIGHT, LOW_CONFIDENCE_BM25_WEIGHT)
                } else {
                    // high vector confidence → use configured weights
                    if args.verbose {
                        println!("\n=== debug: adaptive scoring ===");
                        println!("  top vector score: {:.4}", top_vector_score);
                        println!("  threshold: {:.2}", config.adaptive_threshold);
                        println!("  decision: HIGH CONFIDENCE → use configured weights");
                        println!("  weights: vector={:.2}, bm25={:.2}", config.vector_weight, config.bm25_weight);
                    }
                    (config.vector_weight, config.bm25_weight)
                }
            } else {
                (config.vector_weight, config.bm25_weight)
            };

            // fuse results using weighted sum
            let strategy = FusionStrategy::WeightedSum {
                vector_weight: effective_vector_weight,
                bm25_weight: effective_bm25_weight,
            };
            let fused = fuse_results(&vector_results, &bm25_results, strategy, initial_limit)?;

            tracing::info!("fusion returned {} results", fused.len());

            // fetch text from qdrant for bm25-only results (those with empty text)
            let missing_text_ids: Vec<String> = fused
                .iter()
                .filter(|r| r.text.is_empty())
                .map(|r| r.doc_id.clone())
                .collect();

            let payload_map: std::collections::HashMap<String, (String, String)> = if !missing_text_ids.is_empty() {
                tracing::debug!("fetching text for {} bm25-only results", missing_text_ids.len());
                let point_ids: Vec<qdrant_client::qdrant::PointId> = missing_text_ids
                    .iter()
                    .map(|id| qdrant_client::qdrant::PointId::from(id.clone()))
                    .collect();

                let get_response = qdrant_client
                    .get_points(
                        GetPointsBuilder::new(collection_name.clone(), point_ids)
                            .with_payload(true),
                    )
                    .await?;

                get_response
                    .result
                    .into_iter()
                    .map(|point| {
                        let id = point_id_to_string(point.id);
                        let text = point
                            .payload
                            .get("text")
                            .and_then(|v| value_as_string(v))
                            .unwrap_or_default();
                        let source = point
                            .payload
                            .get("source")
                            .and_then(|v| value_as_string(v))
                            .unwrap_or_default();
                        (id, (text, source))
                    })
                    .collect()
            } else {
                std::collections::HashMap::new()
            };

            // convert fused results to QueryResultOutput
            fused
                .into_iter()
                .map(|result| {
                    // get text and source - prefer payload_map for bm25-only, use result.text for vector
                    let (text, source_path) = if result.text.is_empty() {
                        payload_map
                            .get(&result.doc_id)
                            .cloned()
                            .unwrap_or_default()
                    } else {
                        let src = vector_source_map
                            .get(&result.doc_id)
                            .cloned()
                            .unwrap_or_default();
                        (result.text, src)
                    };

                    let source_raw = if source_path.is_empty() {
                        Some(result.doc_id.clone())
                    } else {
                        Some(source_path)
                    };
                    let citation_raw = format_citation(source_raw.as_deref());
                    let citation = display_citation(&citation_raw, source_raw.as_deref(), CitationMode::Full);
                    let source = display_source(source_raw.as_deref(), CitationMode::Full);

                    QueryResultOutput {
                        id: result.doc_id,
                        score: result.score,
                        text,
                        citation,
                        source,
                        metadata: std::collections::HashMap::new(),
                    }
                })
                .collect()
        }
    };

    // apply reranking if enabled
    let results = if config.enable_rerank && !candidates.is_empty() {
        tracing::info!("reranking {} candidates", candidates.len());

        // load reranker model
        let rerank_model_info = RerankModelInfo::custom(
            config.rerank_model_repo.clone(),
            config.rerank_model_file.clone(),
            512,
            true,
        );
        let rerank_model_path = ensure_rerank_model(&rerank_model_info, &config.model_cache_dir)?;
        let reranker = OnnxReranker::new(
            &rerank_model_path,
            rerank_model_info,
            RerankExecutionProvider::detect(),
        )?;

        // create (query, document) pairs
        let pairs: Vec<(String, String)> = candidates
            .iter()
            .map(|candidate| (query.to_string(), candidate.text.clone()))
            .collect();

        // score with reranker
        let rerank_scores = reranker.score_batch(&pairs)?;

        // combine candidates with rerank scores
        let mut ranked: Vec<(QueryResultOutput, f32)> = candidates
            .into_iter()
            .zip(rerank_scores.into_iter())
            .collect();

        // sort by rerank score (descending)
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // take top rerank_top_k results
        let final_limit = config.rerank_top_k.min(ranked.len());
        let final_results: Vec<QueryResultOutput> = ranked
            .into_iter()
            .take(final_limit)
            .map(|(mut result, rerank_score)| {
                result.score = rerank_score;
                result
            })
            .collect();

        tracing::info!("reranking complete, returning {} results", final_results.len());
        final_results
    } else {
        // no reranking, use original vector search results
        candidates.into_iter().take(config.top_k).collect()
    };

    const DEFAULT_WRAP_WIDTH: usize = 100;

    match output_format {
        OutputFormat::Json => {
            let results_json: Vec<serde_json::Value> = results
                .iter()
                .map(|result| {
                    serde_json::json!({
                        "id": &result.id,
                        "score": result.score,
                        "text": &result.text,
                        "citation": &result.citation,
                        "source": &result.source,
                    })
                })
                .collect();
            let output =
                serde_json::to_string_pretty(&serde_json::json!({ "results": results_json }))?;
            println!("{output}");
        }
        OutputFormat::Text => {
            render_text_results(&results, DEFAULT_WRAP_WIDTH, args.verbose, args.commits);
        }
    }

    Ok(())
}

fn build_query_filter(args: &QueryCommand) -> Option<qdrant_client::qdrant::Filter> {
    use qdrant_client::qdrant::{Condition, Filter};

    let mut conditions = Vec::new();
    if let Some(source_prefix) = &args.source {
        let source_prefix = source_prefix.trim();
        if !source_prefix.is_empty() {
            // use text matching for path filtering
            // works best with path segments (e.g., 'backbone' or 'bookmarks/backbone')
            // rather than full absolute paths
            conditions.push(Condition::matches_text("source", source_prefix.to_string()));
        }
    }

    if conditions.is_empty() {
        None
    } else {
        Some(Filter::must(conditions))
    }
}

fn build_commit_filter(args: &QueryCommand) -> Result<Option<qdrant_client::qdrant::Filter>> {
    use backbone::error::BbtError;
    use qdrant_client::qdrant::{Condition, Filter, Range};
    use chrono::NaiveDate;

    let mut conditions = Vec::new();

    // filter by commit classification
    if let Some(commit_type) = &args.commit_type {
        conditions.push(Condition::matches("classification", commit_type.to_lowercase()));
    }

    // filter by author email
    if let Some(author) = &args.author {
        conditions.push(Condition::matches("author_email", author.to_string()));
    }

    // filter by date range (since)
    if let Some(since_str) = &args.since {
        let date = NaiveDate::parse_from_str(since_str, "%Y-%m-%d")
            .map_err(|e| BbtError::Schema(format!("invalid date format for --since: {} (expected YYYY-MM-DD)", e)))?;
        let since_time = date
            .and_hms_opt(0, 0, 0)
            .ok_or_else(|| BbtError::Schema("invalid time".to_string()))?
            .and_utc();

        conditions.push(Condition::range(
            "commit_time",
            Range {
                gte: Some(since_time.timestamp() as f64),
                ..Default::default()
            },
        ));
    }

    // filter by date range (until)
    if let Some(until_str) = &args.until {
        let date = NaiveDate::parse_from_str(until_str, "%Y-%m-%d")
            .map_err(|e| BbtError::Schema(format!("invalid date format for --until: {} (expected YYYY-MM-DD)", e)))?;
        let until_time = date
            .and_hms_opt(23, 59, 59)
            .ok_or_else(|| BbtError::Schema("invalid time".to_string()))?
            .and_utc();

        conditions.push(Condition::range(
            "commit_time",
            Range {
                lte: Some(until_time.timestamp() as f64),
                ..Default::default()
            },
        ));
    }

    Ok(if conditions.is_empty() {
        None
    } else {
        Some(Filter::must(conditions))
    })
}

fn format_citation(source: Option<&str>) -> String {
    if let Some(source_path) = source {
        let path = Path::new(source_path);
        if let Some(stem) = path.file_stem().and_then(|value| value.to_str()) {
            return stem.to_string();
        }
    }
    "unknown".to_string()
}

fn display_citation(
    citation: &str,
    source: Option<&str>,
    mode: CitationMode,
) -> String {
    match mode {
        CitationMode::Full | CitationMode::Basename => citation.to_string(),
        CitationMode::Hash => hash_label(source.unwrap_or(citation)),
        CitationMode::Redact => "redacted".to_string(),
    }
}

fn display_source(source: Option<&str>, mode: CitationMode) -> String {
    match mode {
        CitationMode::Full => source.unwrap_or("unknown").to_string(),
        CitationMode::Basename => source
            .and_then(|value| Path::new(value).file_name().and_then(|name| name.to_str()))
            .unwrap_or("unknown")
            .to_string(),
        CitationMode::Hash => hash_label(source.unwrap_or("unknown")),
        CitationMode::Redact => "redacted".to_string(),
    }
}

fn hash_label(value: &str) -> String {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    if let Ok(secret) = std::env::var("QUERY_CITATION_SECRET") {
        if !secret.is_empty() {
            hasher.update(secret.as_bytes());
        }
    }
    hasher.update(value.as_bytes());
    let digest = hasher.finalize();
    let digest_hex = format!("{digest:x}");
    let prefix_len = 12.min(digest_hex.len());
    format!("src_{}", &digest_hex[..prefix_len])
}

fn value_as_string(value: &qdrant_client::qdrant::Value) -> Option<String> {
    use qdrant_client::qdrant::value::Kind as ValueKind;

    match &value.kind {
        Some(ValueKind::StringValue(value)) => Some(value.clone()),
        Some(ValueKind::IntegerValue(value)) => Some(value.to_string()),
        Some(ValueKind::DoubleValue(value)) => Some(value.to_string()),
        Some(ValueKind::BoolValue(value)) => Some(value.to_string()),
        _ => None,
    }
}

fn point_id_to_string(id: Option<qdrant_client::qdrant::PointId>) -> String {
    use qdrant_client::qdrant::point_id::PointIdOptions;

    match id.and_then(|value| value.point_id_options) {
        Some(PointIdOptions::Num(value)) => value.to_string(),
        Some(PointIdOptions::Uuid(value)) => value,
        None => "unknown".to_string(),
    }
}

fn render_text_results(results: &[QueryResultOutput], wrap_width: usize, show_scores: bool, is_commits: bool) {
    if results.is_empty() {
        println!("no results");
        return;
    }

    let width = if wrap_width == 0 { 100 } else { wrap_width };
    println!("results: {}", results.len());

    for (index, result) in results.iter().enumerate() {
        println!();
        println!("result {}/{}", index + 1, results.len());
        if show_scores {
            println!("score: {:.4}", result.score);
        }

        if is_commits {
            // commit-specific formatting
            let commit_hash = result.metadata.get("commit_hash")
                .map(|s| &s[..7.min(s.len())])  // short hash (7 chars)
                .unwrap_or("unknown");
            let classification = result.metadata.get("classification")
                .map(|s| s.as_str())
                .unwrap_or("unknown");
            let author_name = result.metadata.get("author_name")
                .map(|s| s.as_str())
                .unwrap_or("unknown");
            let author_email = result.metadata.get("author_email")
                .map(|s| s.as_str())
                .unwrap_or("");
            let commit_time = result.metadata.get("commit_time")
                .map(|s| s.as_str())
                .unwrap_or("unknown");
            let insertions = result.metadata.get("insertions")
                .map(|s| s.as_str())
                .unwrap_or("0");
            let deletions = result.metadata.get("deletions")
                .map(|s| s.as_str())
                .unwrap_or("0");
            let file_paths = result.metadata.get("file_paths")
                .map(|s| s.as_str())
                .unwrap_or("");
            let repo_name = result.metadata.get("repository_name")
                .map(|s| s.as_str())
                .unwrap_or("unknown");

            // format: [hash] type: message first line
            let message_first_line = result.text.lines().next().unwrap_or("");
            println!("[{}] {}: {}", commit_hash, classification, message_first_line);
            println!("repository: {}", repo_name);
            println!("author: {} <{}>", author_name, author_email);
            println!("date: {}", commit_time);
            if file_paths.is_empty() {
                println!("changes: +{} -{}", insertions, deletions);
            } else {
                println!("files: {}", file_paths);
                println!("changes: +{} -{}", insertions, deletions);
            }
            println!();
            println!("message:");
            let wrapped_lines = wrap_text(&result.text, width);
            if wrapped_lines.is_empty() {
                println!("  [empty]");
            } else {
                for line in wrapped_lines {
                    println!("  {}", line);
                }
            }
        } else {
            // document-specific formatting (original)
            let text_chars = result.text.chars().count();
            let wrapped_lines = wrap_text(&result.text, width);

            println!("citation: {}", result.citation);
            println!("source: {}", result.source);
            println!("text_chars: {}", text_chars);
            println!("text:");
            if wrapped_lines.is_empty() {
                println!("  [empty]");
            } else {
                for line in wrapped_lines {
                    println!("  {}", line);
                }
            }
        }

        println!("{}", "-".repeat(40));
    }
}

fn wrap_text(text: &str, width: usize) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }

    let mut lines = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        if line.is_empty() {
            line.push_str(word);
            continue;
        }

        if line.len() + word.len() + 1 > width {
            lines.push(line);
            line = word.to_string();
        } else {
            line.push(' ');
            line.push_str(word);
        }
    }

    if !line.is_empty() {
        lines.push(line);
    }

    lines
}


async fn generate_schema(
    input: PathBuf,
    output: PathBuf,
    json_schema_output: Option<PathBuf>,
    repo_id: Option<String>,
    filename: Option<String>,
    cache_dir: Option<PathBuf>,
) -> Result<()> {
    use backbone::agent::parser::extract_create_table;
    use backbone::agent::prompt::{build_schema_task_prompt, SCHEMA_SYSTEM_PROMPT};
    use backbone::llm::downloader::ensure_gguf_model;
    use backbone::llm::model::{LlamaModel, Message, MessageRole, ModelConfig};
    use backbone::schema::inference::infer_json_schema;
    use backbone::schema::hints::derive_field_hints;
    use std::fs;

    let model_path = ensure_gguf_model(repo_id, filename, cache_dir)?;

    let json_str = fs::read_to_string(&input)?;
    let json_value: serde_json::Value = serde_json::from_str(&json_str)?;

    let schema = infer_json_schema(&json_value);
    let hints = derive_field_hints(&schema);
    let user_prompt = build_schema_task_prompt(&schema, &hints);

    let model_config = ModelConfig::default();
    let model = LlamaModel::new(model_path, model_config)?;

    let messages = vec![
        Message {
            role: MessageRole::System,
            content: SCHEMA_SYSTEM_PROMPT.to_string(),
        },
        Message {
            role: MessageRole::User,
            content: user_prompt,
        },
    ];

    let response = model.generate(messages)?;
    let sql = extract_create_table(&response)?;

    fs::write(&output, sql)?;
    println!("sql schema written to: {}", output.display());

    if let Some(schema_output) = json_schema_output {
        let formatted_json = serde_json::to_string_pretty(&json_value)?;
        fs::write(&schema_output, &formatted_json)?;
        println!("json schema written to: {}", schema_output.display());
    }

    Ok(())
}

async fn generate_json_schema(
    input: PathBuf,
    output: PathBuf,
    _repo_id: Option<String>,
    _filename: Option<String>,
    _cache_dir: Option<PathBuf>,
) -> Result<()> {
    use std::fs;

    let json_str = fs::read_to_string(&input)?;
    let json_value: serde_json::Value = serde_json::from_str(&json_str)?;
    let formatted_json = serde_json::to_string_pretty(&json_value)?;

    fs::write(&output, formatted_json)?;
    println!("JSON schema written to: {}", output.display());

    Ok(())
}
