use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use std::path::{Path, PathBuf};

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
    #[clap(long, env = "BBT_FORCE_SYNC", default_value = "false")]
    force: bool,

    /// Reset the processing state before syncing
    #[clap(long, env = "BBT_RESET_STATE", default_value = "false")]
    reset_state: bool,
}

#[derive(Args)]
pub struct QueryCommand {
    /// Query text
    #[clap(long)]
    query: String,

    /// Override top k results
    #[clap(long)]
    top_k: Option<usize>,

    /// Retrieval mode override
    #[clap(long, value_enum)]
    mode: Option<QueryMode>,

    /// Filter by source path prefix
    #[clap(long)]
    source_prefix: Option<String>,

    /// Citation redaction mode
    #[clap(long, value_enum, default_value = "full")]
    citation_mode: CitationMode,

    /// Output format
    #[clap(long, value_enum, default_value = "text")]
    format: OutputFormat,

    /// Wrap width for text output
    #[clap(long, default_value_t = 100)]
    wrap_width: usize,

    /// Show scores in text output
    #[clap(long, default_value = "false")]
    show_scores: bool,
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
        #[arg(long, env = "BBT_SCHEMA_REPO_ID")]
        repo_id: Option<String>,

        /// Model filename
        #[arg(long, env = "BBT_SCHEMA_FILENAME")]
        filename: Option<String>,

        /// Model cache directory
        #[arg(long, env = "BBT_MODEL_CACHE_DIR")]
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
        #[arg(long, env = "BBT_SCHEMA_REPO_ID")]
        repo_id: Option<String>,

        /// Model filename
        #[arg(long, env = "BBT_SCHEMA_FILENAME")]
        filename: Option<String>,

        /// Model cache directory
        #[arg(long, env = "BBT_MODEL_CACHE_DIR")]
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
                    // bash/shell
                    "sh", "bash", "zsh", "fish",
                    // terraform/hcl
                    "tf", "tfvars", "hcl",
                    // template files
                    "j2", "jinja", "jinja2", "hbs", "handlebars", "mustache",
                    "tmpl", "template", "erb", "ejs", "tpl",
                    // common config/doc files
                    "toml", "yaml", "yml", "json", "md",
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

async fn sync_documents(args: SyncCommand) -> Result<()> {
    use backbone::config::BbtConfig;
    use backbone::document::loader::DocumentLoader;
    use backbone::document::text::chunking::chunk_text;
    use backbone::embedding::{ensure_onnx_model, EmbeddingModelInfo, ExecutionProvider, OnnxEmbedder};
    use backbone::retrieval::bm25::{load_index, save_index};
    use backbone::storage::StateStore;
    use backbone::tracing::init_tracing;
    use indicatif::{HumanBytes, MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
    use console::Term;
    use qdrant_client::Qdrant;
    use qdrant_client::qdrant::{PointStruct, UpsertPointsBuilder, Value};
    use std::fs;
    use std::collections::HashMap;
    use uuid::Uuid;

    let config = BbtConfig::from_env()?;

    let loader = DocumentLoader::new()?;
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

    let term = Term::stderr();
    let term_width = usize::from(term.size().1);
    let progress = MultiProgress::with_draw_target(ProgressDrawTarget::stderr());
    progress.set_move_cursor(true);
    let scan_bar = progress.add(ProgressBar::new_spinner());
    scan_bar.set_style(
        ProgressStyle::with_template("{prefix} {spinner} {msg}")?
            .tick_chars("|/-\\"),
    );
    scan_bar.set_prefix("scan");
    scan_bar.enable_steady_tick(std::time::Duration::from_millis(120));

    let log_bar = progress.add(ProgressBar::new(0));
    log_bar.set_style(ProgressStyle::with_template("{prefix} {msg}")?);
    log_bar.set_prefix("log");
    log_bar.set_message("ready");

    let log_message_width = log_message_width(term_width, log_bar.prefix().len());
    let (log_tx, log_rx) = std::sync::mpsc::channel::<String>();
    let log_bar_handle = log_bar.clone();
    let log_thread = std::thread::spawn(move || {
        while let Ok(line) = log_rx.recv() {
            if line == "__bbt_log_close__" {
                break;
            }
            let message = truncate_log_line(&line, log_message_width);
            log_bar_handle.set_message(message);
        }
    });

    let _guard = init_tracing("bbt", Some(log_tx.clone()))?;

    let scan_scope_key = build_scan_scope_key(&args, &extensions);
    let scan_message_width = scan_message_width(term_width, scan_bar.prefix().len());

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
                    &scan_bar,
                    &mut state_store,
                    &scan_scope_key,
                    &scan_progress,
                    scan_message_width,
                )?;
                state_store.finish_scan(&scan_scope_key, &scan_progress.run_id)?;
                scan_bar.finish_and_clear();
                progress.remove(&scan_bar);
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
                scan_bar.set_message(format!(
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
                        &scan_bar,
                        &mut state_store,
                        &scan_scope_key,
                        &scan_progress,
                        scan_message_width,
                    )?;
                    state_store.finish_scan(&scan_scope_key, &scan_progress.run_id)?;
                    scan_bar.finish_and_clear();
                    progress.remove(&scan_bar);
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
                    scan_bar.finish_and_clear();
                    progress.remove(&scan_bar);
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
            scan_bar.set_message(format!(
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
                &scan_bar,
                &mut state_store,
                &scan_scope_key,
                &scan_progress,
                scan_message_width,
            )?;
            state_store.finish_scan(&scan_scope_key, &scan_progress.run_id)?;
            scan_bar.finish_and_clear();
            progress.remove(&scan_bar);
            tracing::info!(
                "scan complete: {} files, {} bytes",
                scan_totals.files,
                scan_totals.bytes
            );
            (candidates, scan_totals, scan_progress.run_id)
        }
    } else {
        let scan_progress = state_store.start_scan(&scan_scope_key)?;
        scan_bar.set_message(format!(
            "{} files, {} bytes",
            scan_progress.files_seen, scan_progress.bytes_seen
        ));
        let (candidates, scan_totals) = collect_candidates(
            &args,
            &extensions,
            &scan_bar,
            &mut state_store,
            &scan_scope_key,
            &scan_progress,
            scan_message_width,
        )?;
        state_store.finish_scan(&scan_scope_key, &scan_progress.run_id)?;
        scan_bar.finish_and_clear();
        progress.remove(&scan_bar);
        tracing::info!(
            "scan complete: {} files, {} bytes",
            scan_totals.files,
            scan_totals.bytes
        );
        (candidates, scan_totals, scan_progress.run_id)
    };

    let total_files = scan_totals.files;
    let total_bytes = scan_totals.bytes;
    let files_bar = progress.insert_before(&log_bar, ProgressBar::new(total_files));
    let bytes_bar = progress.insert_before(&log_bar, ProgressBar::new(total_bytes));

    let files_style = ProgressStyle::with_template(
        "{prefix} [{bar:40.cyan/blue}] {pos}/{len} {msg}",
    )?;
    let bytes_style = ProgressStyle::with_template(
        "{prefix} [{bar:40.green/green}] {bytes}/{total_bytes} {msg}",
    )?;

    files_bar.set_style(files_style);
    bytes_bar.set_style(bytes_style);
    files_bar.set_prefix("files");
    bytes_bar.set_prefix("bytes");
    let files_count_width = total_files.to_string().len();
    let bytes_count_width = format!("{}", HumanBytes(total_bytes)).len();
    let files_message_width = progress_message_width(
        term_width,
        files_bar.prefix().len(),
        40,
        files_count_width,
        files_count_width,
    );
    let bytes_message_width = progress_message_width(
        term_width,
        bytes_bar.prefix().len(),
        40,
        bytes_count_width,
        bytes_count_width,
    );

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

    files_bar.set_position(resumed_files);
    bytes_bar.set_position(resumed_bytes);

    let model_info = EmbeddingModelInfo::default_model();
    let vector_dimensions = model_info.dimensions;

    // ensure model is downloaded
    let model_path = ensure_onnx_model(&model_info, &config.model_cache_dir)?;
    let embedder = OnnxEmbedder::new(&model_path, model_info, ExecutionProvider::detect())?;

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

    // verify connection to qdrant with retry
    tracing::info!("verifying connection to qdrant at {}", config.qdrant_url);
    let mut retry_count = 0;
    let max_retries = 5;
    let mut connected = false;

    while retry_count <= max_retries && !connected {
        match qdrant_client.health_check().await {
            Ok(_) => {
                tracing::info!("qdrant connection verified");
                connected = true;
            }
            Err(e) => {
                if retry_count < max_retries {
                    let delay_ms = 500 * (2_u64.pow(retry_count.min(3)));
                    tracing::warn!(
                        "connection attempt {}/{} failed, retrying in {}ms: {}",
                        retry_count + 1,
                        max_retries + 1,
                        delay_ms,
                        e
                    );
                    tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
                    retry_count += 1;
                } else {
                    tracing::error!("failed to connect to qdrant after {} attempts: {}", max_retries + 1, e);
                    anyhow::bail!("cannot connect to qdrant at {}: {}", config.qdrant_url, e);
                }
            }
        }
    }

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

    tracing::info!(
        "starting sync for paths: {:?}, with extensions: {:?}, git_only={}, honor_gitignore={}",
        args.paths,
        extensions,
        args.git_only,
        args.honor_gitignore
    );

    for candidate in candidates {
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

        // skip empty files silently (e.g., __init__.py)
        if file_size_bytes == 0 {
            tracing::debug!("skipping empty file {:?}", file_path);
            files_bar.inc(1);
            // note: not incrementing bytes_bar since size is 0
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
                let files_message = right_align_truncate(&filename, files_message_width);
                let bytes_message = right_align_truncate(&filename, bytes_message_width);
                files_bar.set_message(files_message);
                bytes_bar.set_message(bytes_message);
                files_bar.inc(1);
                bytes_bar.inc(candidate.size_bytes);
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
                let files_message = right_align_truncate(&filename, files_message_width);
                let bytes_message = right_align_truncate(&filename, bytes_message_width);
                files_bar.set_message(files_message);
                bytes_bar.set_message(bytes_message);
                files_bar.inc(1);
                bytes_bar.inc(candidate.size_bytes);
                continue;
            }
        };

        if chunks.is_empty() {
            tracing::warn!(
                "skipping file {:?} because no chunks were produced",
                file_path
            );
            let _ = state_store.record_failure(
                &source_path,
                Some(doc.file_hash.clone()),
                file_size_bytes,
                modified_at,
                "no chunks produced".to_string(),
            );
            let _ = state_store.mark_failed(&source_path, "no chunks produced".to_string());
            let files_message = right_align_truncate(&filename, files_message_width);
            let bytes_message = right_align_truncate(&filename, bytes_message_width);
            files_bar.set_message(files_message);
            bytes_bar.set_message(bytes_message);
            files_bar.inc(1);
            bytes_bar.inc(candidate.size_bytes);
            continue;
        }

        let embeddings = match embedder.embed(&chunks) {
            Ok(em) => em,
            Err(e) => {
                tracing::error!(
                    "failed to embed chunks for {:?}: {}",
                    file_path,
                    e
                );
                let _ = state_store.record_failure(
                    &source_path,
                    Some(doc.file_hash.clone()),
                    file_size_bytes,
                    modified_at,
                    format!("embedding failed: {e}"),
                );
                let _ = state_store.mark_failed(&source_path, e.to_string());
                let files_message = right_align_truncate(&filename, files_message_width);
                let bytes_message = right_align_truncate(&filename, bytes_message_width);
                files_bar.set_message(files_message);
                bytes_bar.set_message(bytes_message);
                files_bar.inc(1);
                bytes_bar.inc(candidate.size_bytes);
                continue;
            }
        };

        if embeddings.is_empty() {
            tracing::warn!(
                "skipping file {:?} because embeddings are empty",
                file_path
            );
            let _ = state_store.record_failure(
                &source_path,
                Some(doc.file_hash.clone()),
                file_size_bytes,
                modified_at,
                "embeddings empty".to_string(),
            );
            let _ = state_store.mark_failed(&source_path, "embeddings empty".to_string());
            let files_message = right_align_truncate(&filename, files_message_width);
            let bytes_message = right_align_truncate(&filename, bytes_message_width);
            files_bar.set_message(files_message);
            bytes_bar.set_message(bytes_message);
            files_bar.inc(1);
            bytes_bar.inc(candidate.size_bytes);
            continue;
        }

        if embeddings.len() != chunks.len() {
            tracing::error!(
                "embedding count mismatch for {:?}: chunks={}, embeddings={}",
                file_path,
                chunks.len(),
                embeddings.len()
            );
            let _ = state_store.record_failure(
                &source_path,
                Some(doc.file_hash.clone()),
                file_size_bytes,
                modified_at,
                "embedding count mismatch".to_string(),
            );
            let _ =
                state_store.mark_failed(&source_path, "embedding count mismatch".to_string());
            let files_message = right_align_truncate(&filename, files_message_width);
            let bytes_message = right_align_truncate(&filename, bytes_message_width);
            files_bar.set_message(files_message);
            bytes_bar.set_message(bytes_message);
            files_bar.inc(1);
            bytes_bar.inc(candidate.size_bytes);
            continue;
        }

        // create points and add to bm25 index
        let mut points = Vec::new();
        for (chunk_text, embedding) in chunks.into_iter().zip(embeddings.into_iter()) {
            // generate unique id for chunk
            let chunk_id = Uuid::new_v4().to_string();

            // create qdrant point
            let mut payload: HashMap<String, Value> = HashMap::new();
            payload.insert("text".to_string(), chunk_text.clone().into());
            payload.insert(
                "source".to_string(),
                file_path.to_string_lossy().to_string().into(),
            );
            points.push(PointStruct::new(chunk_id.clone(), embedding, payload));

            // add to bm25 index
            if let Err(e) = bm25_index.add_document(&chunk_id, &chunk_text) {
                tracing::warn!("failed to add chunk {} to bm25 index: {}", chunk_id, e);
            }
        }

        // retry upsert with exponential backoff
        let mut retry_count = 0;
        let max_retries = 3;
        let mut success = false;

        while retry_count <= max_retries && !success {
            let upsert_request =
                UpsertPointsBuilder::new(collection_name.clone(), points.clone()).build();
            match qdrant_client.upsert_points(upsert_request).await {
                Ok(_) => {
                    tracing::debug!("successfully ingested file {:?}", file_path);
                    let _ = state_store.mark_complete(&source_path, points.len());
                    let _ = state_store.clear_failure(&source_path);
                    let files_message = right_align_truncate(&filename, files_message_width);
                    let bytes_message = right_align_truncate(&filename, bytes_message_width);
                    files_bar.set_message(files_message);
                    bytes_bar.set_message(bytes_message);
                    files_bar.inc(1);
                    bytes_bar.inc(candidate.size_bytes);
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
                            Some(doc.file_hash.clone()),
                            file_size_bytes,
                            modified_at,
                            format!("upsert failed: {e}"),
                        );
                        let _ = state_store.mark_failed(&source_path, e.to_string());
                        let files_message = right_align_truncate(&filename, files_message_width);
                        let bytes_message = right_align_truncate(&filename, bytes_message_width);
                        files_bar.set_message(files_message);
                        bytes_bar.set_message(bytes_message);
                        files_bar.inc(1);
                        bytes_bar.inc(candidate.size_bytes);
                        retry_count += 1;
                    }
                }
            }
        }
    }

    files_bar.finish_with_message("done");
    bytes_bar.finish_with_message("done");

    // save bm25 index
    tracing::info!("saving bm25 index with {} documents", bm25_index.num_docs());
    if let Err(e) = save_index(&bm25_index, &config.bm25_index_path) {
        tracing::error!("failed to save bm25 index: {}", e);
    }

    tracing::info!("sync completed");
    let _ = log_tx.send("__bbt_log_close__".to_string());
    drop(log_tx);
    let _ = log_thread.join();
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
    scan_bar: &indicatif::ProgressBar,
    state_store: &mut backbone::storage::StateStore,
    scan_scope_key: &str,
    scan_progress: &backbone::storage::ScanProgress,
    scan_message_width: usize,
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
                scan_bar.set_message(format_scan_message(
                    totals.files,
                    totals.bytes,
                    &source_path,
                    scan_message_width,
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

    scan_bar.set_message(format!(
        "{} files, {} bytes",
        totals.files, totals.bytes
    ));

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

fn log_message_width(term_width: usize, prefix_len: usize) -> usize {
    let left_width = prefix_len.saturating_add(1);
    term_width.saturating_sub(left_width).saturating_sub(1)
}

fn truncate_log_line(value: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let value_chars: Vec<char> = value.chars().collect();
    if value_chars.len() <= width {
        return pad_right(value, width);
    }
    if width <= 3 {
        return ".".repeat(width);
    }
    let available = width - 3;
    let prefix_len = (available + 1) / 2;
    let suffix_len = available - prefix_len;
    let prefix = value_chars[..prefix_len].iter().collect::<String>();
    let suffix = value_chars[value_chars.len() - suffix_len..]
        .iter()
        .collect::<String>();
    pad_right(&format!("{prefix}...{suffix}"), width)
}

fn pad_right(value: &str, width: usize) -> String {
    let length = value.chars().count();
    if length >= width {
        value.to_string()
    } else {
        format!("{}{}", value, " ".repeat(width - length))
    }
}

fn progress_message_width(
    term_width: usize,
    prefix_len: usize,
    bar_width: usize,
    left_value_width: usize,
    right_value_width: usize,
) -> usize {
    let fixed = prefix_len
        .saturating_add(1) // space after prefix
        .saturating_add(1) // [
        .saturating_add(bar_width)
        .saturating_add(1) // ]
        .saturating_add(1) // space
        .saturating_add(left_value_width)
        .saturating_add(1) // /
        .saturating_add(right_value_width)
        .saturating_add(1); // space before msg
    term_width.saturating_sub(fixed)
}

fn scan_message_width(term_width: usize, prefix_len: usize) -> usize {
    let spinner_len = 1_usize;
    let left_width = prefix_len.saturating_add(spinner_len).saturating_add(2);
    term_width.saturating_sub(left_width).saturating_sub(1)
}

fn format_scan_message(
    files: u64,
    bytes: u64,
    source_path: &str,
    width: usize,
) -> String {
    let counts = format!("{} files {} bytes", files, bytes);
    if width == 0 {
        return counts;
    }

    let spacer = " ";
    let available = width.saturating_sub(counts.len()).saturating_sub(spacer.len());
    let file_display = right_align_truncate(source_path, available);
    format!("{counts}{spacer}{file_display}")
}

fn right_align_truncate(value: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }

    let value_chars: Vec<char> = value.chars().collect();
    let value_len = value_chars.len();

    let truncated = if value_len > width {
        if width <= 3 {
            ".".repeat(width)
        } else {
            let suffix_len = width - 3;
            let suffix = value_chars[value_len - suffix_len..].iter().collect::<String>();
            format!("...{suffix}")
        }
    } else {
        value.to_string()
    };

    let truncated_len = truncated.chars().count();
    if truncated_len >= width {
        truncated
    } else {
        format!("{}{}", " ".repeat(width - truncated_len), truncated)
    }
}

async fn query_documents(args: QueryCommand) -> Result<()> {
    use backbone::config::{BbtConfig, RetrievalMode};
    use backbone::embedding::{ensure_onnx_model, EmbeddingModelInfo, OnnxEmbedder};
    use backbone::embedding::ExecutionProvider as EmbeddingExecutionProvider;
    use backbone::reranking::{ensure_rerank_model, ExecutionProvider as RerankExecutionProvider, OnnxReranker, RerankModelInfo};
    use backbone::retrieval::{Bm25Scorer, FusionStrategy, fuse_results};
    use backbone::retrieval::bm25::load_index;
    use backbone::tracing::init_tracing;
    use qdrant_client::qdrant::SearchPointsBuilder;
    use qdrant_client::Qdrant;

    let _guard = init_tracing("bbt", None)?;
    let mut config = BbtConfig::from_env()?;

    if let Some(mode) = args.mode {
        config.retrieval_mode = mode.into();
    }
    if let Some(top_k) = args.top_k {
        config.top_k = top_k;
    }

    let query = args.query.trim();
    if query.is_empty() {
        anyhow::bail!("query is required");
    }

    if config.top_k == 0 {
        anyhow::bail!("top_k must be greater than 0");
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

            let model_info = EmbeddingModelInfo::default_model();
            let model_path = ensure_onnx_model(&model_info, &config.model_cache_dir)?;
            let embedder = OnnxEmbedder::new(&model_path, model_info, EmbeddingExecutionProvider::detect())?;

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

            let collection_name = "bbt".to_string();
            let collection_exists = qdrant_client
                .collection_exists(&collection_name)
                .await
                .unwrap_or(false);
            if !collection_exists {
                anyhow::bail!("collection '{}' not found, run sync first", collection_name);
            }

            let mut search_builder =
                SearchPointsBuilder::new(collection_name.clone(), query_vector, initial_limit as u64)
                    .with_payload(true);

            if let Some(filter) = build_query_filter(&args) {
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
                let citation = display_citation(&citation_raw, source_raw.as_deref(), args.citation_mode);
                let source = display_source(source_raw.as_deref(), args.citation_mode);
                let id = point_id_to_string(point.id);

                results.push(QueryResultOutput {
                    id,
                    score: point.score,
                    text,
                    citation,
                    source,
                });
            }
            results
        }
        RetrievalMode::Bm25 => {
            tracing::info!("bm25 search with top_k={}", initial_limit);

            let index = bm25_index.as_ref().unwrap();
            let scorer = Bm25Scorer::with_params(config.bm25_k1, config.bm25_b);
            let bm25_results = scorer.search(index, query, initial_limit);

            tracing::info!("bm25 returned {} results", bm25_results.len());

            // convert bm25 results to QueryResultOutput format
            bm25_results
                .into_iter()
                .map(|result| {
                    let source_raw = index
                        .get_document(&result.doc_id)
                        .and_then(|_doc| {
                            // extract source from chunk_text metadata if available
                            // for now, use doc_id as fallback
                            Some(result.doc_id.clone())
                        });
                    let citation_raw = format_citation(source_raw.as_deref());
                    let citation = display_citation(&citation_raw, source_raw.as_deref(), args.citation_mode);
                    let source = display_source(source_raw.as_deref(), args.citation_mode);

                    QueryResultOutput {
                        id: result.doc_id,
                        score: result.score,
                        text: result.text,
                        citation,
                        source,
                    }
                })
                .collect()
        }
        RetrievalMode::Hybrid => {
            tracing::info!("hybrid search (vector + bm25) with top_k={}", initial_limit);

            // perform vector search
            let model_info = EmbeddingModelInfo::default_model();
            let model_path = ensure_onnx_model(&model_info, &config.model_cache_dir)?;
            let embedder = OnnxEmbedder::new(&model_path, model_info, EmbeddingExecutionProvider::detect())?;

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

            // convert vector results
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

            // fuse results using weighted sum
            let strategy = FusionStrategy::WeightedSum {
                vector_weight: config.vector_weight,
                bm25_weight: config.bm25_weight,
            };
            let fused = fuse_results(&vector_results, &bm25_results, strategy, initial_limit)?;

            tracing::info!("fusion returned {} results", fused.len());

            // convert fused results to QueryResultOutput
            fused
                .into_iter()
                .map(|result| {
                    let source_raw = Some(result.doc_id.clone());
                    let citation_raw = format_citation(source_raw.as_deref());
                    let citation = display_citation(&citation_raw, source_raw.as_deref(), args.citation_mode);
                    let source = display_source(source_raw.as_deref(), args.citation_mode);

                    QueryResultOutput {
                        id: result.doc_id,
                        score: result.score,
                        text: result.text,
                        citation,
                        source,
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

    match args.format {
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
            render_text_results(&results, args.wrap_width, args.show_scores);
        }
    }

    Ok(())
}

fn build_query_filter(args: &QueryCommand) -> Option<qdrant_client::qdrant::Filter> {
    use qdrant_client::qdrant::{Condition, Filter};

    let mut conditions = Vec::new();
    if let Some(source_prefix) = &args.source_prefix {
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
    if let Ok(secret) = std::env::var("BBT_CITATION_SECRET") {
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

fn render_text_results(results: &[QueryResultOutput], wrap_width: usize, show_scores: bool) {
    if results.is_empty() {
        println!("no results");
        return;
    }

    let width = if wrap_width == 0 { 100 } else { wrap_width };
    println!("results: {}", results.len());
    for (index, result) in results.iter().enumerate() {
        let text_chars = result.text.chars().count();
        let wrapped_lines = wrap_text(&result.text, width);

        println!();
        println!("result {}/{}", index + 1, results.len());
        if show_scores {
            println!("score: {:.4}", result.score);
        }
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
